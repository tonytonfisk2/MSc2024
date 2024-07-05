from typing import Optional, Tuple, Set
import torch
import torch.nn.functional as F
from torch import nn
import math
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention, TransformerBlock, Transformer, DistilBertPreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class iMultiHeadSelfAttention(MultiHeadSelfAttention):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
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
        #q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        
        scores = 0
        if self.distance_metric == 'cosine_distance':
            scores += torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        elif self.distance_metric == 'manhattan_distance':
            scores += torch.cdist(q, k, p = 1)
        
        scores = scores / math.sqrt(dim_per_head)
        
        #shift score
        if self.alpha > 0:
            scores -= self.alpha
        #Center
        else:
            scores -= torch.mean(scores, dim = -1, keepdim = True)
            
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        #TODO masking for relu function relu(v - score)

        if self.activation_function == 'softmax':
            weights = F.softmax(scores, dim=-1) # (bs, n_heads, q_length, k_length)
            weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)
            # Mask heads if we want to
            if head_mask is not None:
                weights = weights * head_mask
            context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        elif self.activation_function == 'relu':
            #Signed inhibitor or unsigned
            scores  = F.relu(scores)
            #scores = self.dropout(scores)  # (bs, n_heads, q_length, k_length)
            # Mask heads if we want to
            if head_mask is not None:
                weights = weights * head_mask
            if self.signed_inhibitor:
                pos_v = torch.nn.functional.relu(v)
                neg_v = -torch.nn.functional.relu(-v)
                v_sum = torch.sum(v, dim = 2)
                dist1 = torch.cdist(pos_v, scores, p = 1)
                dist2 = torch.cdist(neg_v, -scores, p = 1)
                context = (v_sum + dist1 - dist2) * 0.5
            else:
                v_sum = torch.sum(v, dim = 2)
                z_sum = torch.sum(scores, dim = -1)
                dist = torch.cdist(v, scores, p = 1)
                context = (v_sum - z_sum + dist) * 0.5
        
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)
        
        if output_attentions:
            return (context, weights)
        else:
            return (context,)

class InhibitorTransformerBlock(TransformerBlock):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = iMultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

class InhibitorTransformer(Transformer):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([InhibitorTransformerBlock(config) for _ in range(config.n_layers)])
        self.gradient_checkpointing = False

class InhibitorDistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = InhibitorTransformer(config)  # Encoder

        # Initialize weights and apply final processing
        self.post_init()
        
class InhibitorDistilBertForMaskedLM(DistilBertPreTrainedModel):
    _tied_weights_keys = ["vocab_projector.weight"]

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.activation = get_activation(config.activation)

        self.distilbert = InhibitorDistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()
        
class iDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = InhibitorDistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

