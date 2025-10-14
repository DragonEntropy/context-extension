import torch
from torch import nn
import math
from typing import Callable, Optional

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.utils.generic import TransformersKwargs


class LlamaALiBiConfig(LlamaConfig):
    def __init__(self, alibi=True, **kwargs):
        super().__init__(**kwargs)
        self.alibi = alibi


# Adapted from transformers LlamaAttention
class LlamaAttentionALiBi(LlamaAttention):
    def __init__(self, config: LlamaALiBiConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.register_buffer(
            "slopes",
            self.precompute_slopes(config.num_attention_heads),
            persistent=False,
        )

    def slope_formula(i, num_heads):
        """
        Llama2 7b has 32 attention heads
        ALiBi paper uses geometric formula for slope:
            a = 2^(-8/n)
            r = 2^(-8/n)
        Equivalent formula: 2^(-8(i + 1)/n) for head i
        """
        return math.pow(2, - 8 * (i + 1) / num_heads)

    def precompute_slopes(num_heads):
        # Calculates slops for each attention head
        return torch.tensor([LlamaAttentionALiBi.slope_formula(i, num_heads) for i in range(num_heads)])

    def calculate_alibi(slopes, query_len, key_len):
        device = slopes.device

        # Slopes are scaled by relative distances of each token
        key_ids = torch.arange(key_len).unsqueeze(0)
        query_ids = torch.arange(query_len).unsqueeze(1)
        # 1 x m - n x 1 -> n x m
        relative_ids = torch.clamp(key_ids - query_ids, max=0)
        """
        relative_ids looks something like this:
         0   0   0   0   >
        -1   0   0   0   >
        -2  -1   0   0   >
        -3  -2  -1   0   >
         v   v   v   v
        """
        alibi = (slopes.view(1, -1, 1, 1) * relative_ids.view(1, 1, query_len, key_len)).to(device)
        return alibi

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
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_len = query_states.shape[-2]
        key_len = key_states.shape[-2]

        # Removed application of rotary embeddings here

        if past_key_values is not None:
            # Removed sin and cos from cache_kwargs
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        alibi = None
        if self.config.alibi:
            # ALiBi application to attention
            attention_interface: Callable = LlamaAttentionALiBi.alibi_attention_forward
            alibi = LlamaAttentionALiBi.calculate_alibi(self.slopes, query_len, key_len)
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            alibi=alibi,
            **kwargs
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaALiBiForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaALiBiConfig):
        super().__init__(config)
        if config.alibi:
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = LlamaAttentionALiBi(config, i)
