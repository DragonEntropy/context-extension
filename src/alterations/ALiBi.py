
import torch
from torch import nn
from typing import Callable, Optional
from functorch.experimental.control_flow import map

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv, eager_attention_forward
from transformers.utils.generic import TransformersKwargs


class LlamaALiBiConfig(LlamaConfig):
    def __init__(self, alibi=True, **kwargs):
        super().__init__(**kwargs)
        self.alibi = alibi


# Adapted from transformers LlamaAttention
class LlamaAttentionALiBi(LlamaAttention):
    _NEG_INF = -1e9

    def __init__(self, config: LlamaALiBiConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.register_buffer(
            "slopes",
            LlamaAttentionALiBi.precompute_slopes(config.num_attention_heads),
            persistent=False,
        )

    def precompute_slopes(num_heads):
        """
        Llama2 7b has 32 attention heads
        ALiBi paper uses geometric formula for slope:
            a = 2^(-8/n)
            r = 2^(-8/n)
        Equivalent formula: 2^(-8(i + 1)/n) for head i
        """
        indices = torch.linspace(0, num_heads - 1, num_heads)
        exponents = - 8.0 * (indices + 1.0) / num_heads / float(num_heads)
        return torch.pow(2.0, exponents)

        # Old probably slower: return torch.tensor(math.pow(2, - 8 * (i + 1) / num_heads) for i in range(num_heads))

    def calculate_alibi(slopes, query_len, key_len, offset=0):
        device = slopes.device

        # Slopes are scaled by relative distances of each token
        key_ids = torch.arange(key_len).unsqueeze(0).to(device)
        query_ids = torch.arange(query_len).unsqueeze(1).to(device)
        # 1 x m - n x 1 -> n x m
        relative_ids = torch.where(key_ids <= query_ids + offset, 1.0 * (key_ids - query_ids - offset), - LlamaAttentionALiBi._NEG_INF)
        # relative_ids = torch.where(key_ids <= query_ids + offset, 0, LlamaAttentionALiBi._NEG_INF)
        """
        relative_ids looks something like this (-i: -infinity):
         0  -i  -i  -i   >
        -1   0  -i  -i   >
        -2  -1   0  -i   >
        -3  -2  -1   0   >
         v   v   v   v
        When query_len = 1, offset = k:
        -k  ...  0  -i
        """
        alibi = (slopes.view(1, -1, 1, 1) * relative_ids.view(1, 1, query_len, key_len)).to(device, dtype=torch.bfloat16)
        # print(f"Query length: {query_len}, Key length: {key_len}, ALiBi shape: {alibi.shape}")
        return alibi
    

    # Eager attention implementation. Unused since llama model uses sdpa
    def alibi_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        alibi: Optional[torch.Tensor] = None,
        **kwargs: TransformersKwargs,
    ):
        # Need custom attention forward code since default applies softmax before ALiBi can be applied
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights += alibi
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Application of RoPE removed here!

        # Cache offset need to correctly compute alibi
        cache_offset = 0
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # past_key_values is a DynamicCache object, a subclass of the Cache object
            cache_offset = past_key_value.get_seq_length(self.layer_idx) - query_states.shape[-2]
        
        query_len = query_states.shape[-2]
        key_len = key_states.shape[-2]

        alibi = None
        attention_interface: Callable = eager_attention_forward
        if self.config.alibi:
            # ALiBi application to attention
            # attention_interface = LlamaAttentionALiBi.alibi_attention_forward_sdpa
            alibi = LlamaAttentionALiBi.calculate_alibi(self.slopes, query_len, key_len, offset=cache_offset).to(query_states.dtype)
            attention_mask = attention_mask + alibi if attention_mask else alibi
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        """
        print(f"After shapes: k: {key_states.shape}, q: {query_states.shape}")
        print(attention_mask)
        print(f"Attention mask shape: {attention_mask.shape}, ALiBi shape: {alibi.shape}")
        """


        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaALiBiForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaALiBiConfig):
        super().__init__(config)
        if config.alibi:
            for i, layer in enumerate(self.model.layers):
                # Need to also copy attention weights
                old_attn = layer.self_attn
                layer.self_attn = LlamaAttentionALiBi(config, i)
                layer.self_attn.load_state_dict(old_attn.state_dict())
