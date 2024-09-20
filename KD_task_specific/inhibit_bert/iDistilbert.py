from typing import Optional, Tuple, Set
import torch
import torch.nn.functional as F
from torch import nn
import math
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention, TransformerBlock, Transformer, DistilBertPreTrainedModel, Embeddings, FFN, DistilBertForSequenceClassification, DistilBertModel
from transformers.configuration_utils import PretrainedConfig

class iMultiHeadSelfAttention(MultiHeadSelfAttention):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.is_causal = False
        # Have an even number of multi heads that divide the dimensions
        if self.dim % self.n_heads != 0:
            # Raise value errors for even multi-head attention nodes
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads: Set[int] = set()
        self.attention_head_size = self.dim // self.n_heads

        self.distance_metric = config.distance_metric
        self.activation_function = config.activation_function
        self.signed_inhibitor = config.signed_inhibitor
        self.alpha = config.alpha # Shift value
        self.center = config.center # Shift center

    def compute_softmax_attention(self, scores: torch.Tensor, mask: torch.Tensor, 
                                  head_mask: Optional[torch.Tensor], v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (mask == 0).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        if head_mask is not None:
            weights = weights * head_mask
        
        context = torch.matmul(weights, v)
        return context, weights

    def compute_relu_attention(self, scores: torch.Tensor, mask: torch.Tensor, 
                               head_mask: Optional[torch.Tensor], v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = scores
        M = torch.max(torch.maximum(v, -v))
        mask = self.dropout(mask.to(scores.dtype))
        mask = (mask == 0).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask, M)
        v_t = v.transpose(-1, -2)
        
        if head_mask is not None:
            weights = weights * head_mask
        
        if self.signed_inhibitor:
            context = self.compute_signed_inhibitor(scores, v, v_t)
        else:
            context = self.compute_unsigned_inhibitor(scores, v, v_t)
        
        return context, weights

    def compute_signed_inhibitor(self, scores: torch.Tensor, v: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        pos_v = F.relu(v_t)
        neg_v = -F.relu(-v_t)
        v_sum = torch.sum(v, dim=-2, keepdim=True)
        dist1 = torch.cdist(scores, pos_v, p=1)
        dist2 = torch.cdist(scores, -neg_v, p=1)
        return 0.5 * (v_sum + dist1 - dist2)

    def compute_unsigned_inhibitor(self, scores: torch.Tensor, v: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        v_sum = torch.sum(v, dim=-2, keepdim=True)
        z_sum = torch.sum(scores, dim=-1, keepdim=True)
        abs_diff = torch.cdist(scores, v_t, p=1)
        return 0.5 * (v_sum - z_sum + abs_diff)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
        
        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        
        if self.distance_metric == 'cosine_distance':
            q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
            scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        elif self.distance_metric == 'manhattan_distance':
            scores = torch.cdist(q, k, p=1) / dim_per_head # Scale the distances
        
        #shift score
        if self.alpha > 0:
            scores -= self.alpha
            scores = F.relu(scores)
            
        #Center
        if self.center:
            scores -= torch.mean(scores, dim = -1, keepdim = True)
            scores = F.relu(scores)

        if self.activation_function == 'softmax':
            context, weights = self.compute_softmax_attention(scores, mask, head_mask, v)
        elif self.activation_function == 'relu':
            context, weights = self.compute_relu_attention(scores, mask, head_mask, v)
        
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)
        
        if output_attentions:
            return (context, weights)
        else:
            return (context,)

class iTransformerBlock(TransformerBlock):
    def __init__(self, config: PretrainedConfig):
        #super().__init__(config)
        nn.Module.__init__(self)

        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = iMultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

class iTransformer(Transformer):
    def __init__(self, config: PretrainedConfig):
        #super().__init__(config)
        nn.Module.__init__(self)
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([iTransformerBlock(config) for _ in range(config.n_layers)])
        self.gradient_checkpointing = False

class iDistilBertModel(DistilBertModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
       
        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = iTransformer(config)  # Encoder

        # Initialize weights and apply final processing
        self.post_init()
        
class iDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = iDistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_weights_except_q_k(self):
        for name, param in self.named_parameters():
            if 'q_lin' not in name and 'k_lin' not in name:
                param.requires_grad = False

    def unfreeze_all_weights(self):
        for param in self.parameters():
            param.requires_grad = True

    

