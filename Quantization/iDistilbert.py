from typing import Optional, Tuple, Set, Union
import torch
import torch.nn.functional as F
from torch import nn
import math
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention, TransformerBlock, Transformer, DistilBertPreTrainedModel, Embeddings, FFN, DistilBertForSequenceClassification, DistilBertModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput, MaskedLMOutput
from dataclasses import dataclass
from transformers.activations import get_activation

#output_contexts = True to return context

@dataclass
class CustomMaskedLMOutput(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    contexts: Optional[Tuple[torch.FloatTensor, ...]] = None
    
@dataclass
class CustomSequenceClassifierOutput(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    contexts: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class CustomModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contexts: Optional[Tuple[torch.FloatTensor]] = None

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
        alpha_init = getattr(config, 'alpha', 0.5)
        beta_init = getattr(config, 'beta', 1.0)
        gamma_init = getattr(config, 'gamma', 1.0)
        #self.alpha = config.alpha # Shift value
        self.center = config.center # Shift center
        #self.gamma = config.gamma # Scale
        #self.beta = config.beta # Scale H
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.get_default_dtype()))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.get_default_dtype()))
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.get_default_dtype()))

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
        scores = F.relu(scores)
        weights = scores
        M = torch.max(torch.maximum(v, -v)) + 1
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
        output_contexts: bool = False,
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
            scale = self.gamma/math.sqrt(dim_per_head)
            scores = torch.cdist(q, k, p=1) * scale
        
        #Center
        if self.center:
            mean = torch.mean(scores, dim = -1, keepdim = True)
            scores -= mean
            
        #shift score
        if self.alpha > 0:
            scores += self.alpha     

        if self.activation_function == 'softmax':
            context, weights = self.compute_softmax_attention(scores, mask, head_mask, v)
        elif self.activation_function == 'relu':
            context, weights = self.compute_relu_attention(scores, mask, head_mask, v)
            context *= self.beta #scale context
        
        context = unshape(context)  # (bs, q_length, dim)
        output = self.out_lin(context) # (bs, q_length, dim)
        
        if output_attentions:
            return (output, weights)
        elif output_contexts:
            return (output, context)
        else:
            return (output,)

class iTransformerBlock(TransformerBlock):
    def __init__(self, config: PretrainedConfig):
        #super().__init__(config)
        nn.Module.__init__(self)
        self.config = config
        if not hasattr(self.config, 'output_contexts'):
            self.config.output_contexts = False
        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = iMultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_contexts: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_contexts= output_contexts,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        elif output_contexts:
            sa_output, sa_context = sa_output  # (bs, seq_length, dim), # (bs, seq_length, dim)
        else:
            assert isinstance(sa_output, tuple), f"sa_output must be a tuple but it is {type(sa_output)} type"
            sa_output = sa_output[0]

        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        outputs = (ffn_output,)
        if output_attentions:
            outputs = (sa_weights,) + outputs 
        elif output_contexts:
            outputs = (sa_context,) + outputs
        return outputs

class iTransformer(Transformer):
    def __init__(self, config: PretrainedConfig):
        #super().__init__(config)
        nn.Module.__init__(self)
        self.config = config
        if not hasattr(self.config, 'output_contexts'):
            self.config.output_contexts = False
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([iTransformerBlock(config) for _ in range(config.n_layers)])
        self.gradient_checkpointing = False
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_contexts: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[CustomModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_contexts = () if output_contexts else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                    output_contexts,
                )
            else:
                layer_outputs = layer_module(
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                    output_contexts,
                )

            hidden_state = layer_outputs[-1]


            if output_attentions:
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
                
            if output_contexts:
                context = layer_outputs[0]  
                all_contexts = all_contexts + (context,)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions, all_contexts] if v is not None)
        return CustomModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions, contexts = all_contexts
        )

class iDistilBertModel(DistilBertModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        if not hasattr(self.config, 'output_contexts'):
            self.config.output_contexts = False
        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = iTransformer(config)  # Encoder

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_contexts: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[CustomModelOutput, Tuple[torch.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_contexts = output_contexts if output_contexts is not None else self.config.output_contexts

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_contexts = output_contexts,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        
class iDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super(DistilBertPreTrainedModel, self).__init__(config)
        if not hasattr(self.config, 'output_contexts'):
            self.config.output_contexts = False
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = iDistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_contexts: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[CustomSequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_contexts=output_contexts,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return CustomSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
            contexts=distilbert_output.contexts if hasattr(distilbert_output, 'contexts') else None,
        )
    def freeze_weights_except_q_k(self):
        for name, param in self.named_parameters():
            if 'q_lin' not in name and 'k_lin' not in name:
                param.requires_grad = False

    def unfreeze_all_weights(self):
        for param in self.parameters():
            param.requires_grad = True


class iDistilBertForMaskedLM(DistilBertPreTrainedModel):
    _tied_weights_keys = ["vocab_projector.weight"]

    def __init__(self, config: PretrainedConfig):
        super(DistilBertPreTrainedModel, self).__init__(config)
        
        if not hasattr(self.config, 'output_contexts'):
            self.config.output_contexts = False
            
        self.activation = get_activation(config.activation)

        self.distilbert = iDistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.vocab_projector

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.vocab_projector = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_contexts: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[CustomMaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_contexts=output_contexts,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return CustomMaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
            contexts=dlbrt_output.contexts if hasattr(dlbrt_output, 'contexts') else None,
        )


    

