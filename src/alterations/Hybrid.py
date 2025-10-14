import torch
from typing import Callable, Optional

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from alterations.ALiBi import LlamaAttentionALiBi, LlamaALiBiConfig


class LlamaHybridConfig(LlamaALiBiConfig):
    def __init__(self, hybrid=True, **kwargs):
        super().__init__(alibi=hybrid, **kwargs)
        self.hybrid = hybrid


# Adapted from transformers LlamaAttention
class LlamaAttentionHybrid(LlamaAttentionALiBi):
    def __init__(self, config: LlamaHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)

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

        # Using original RoPE definition
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        alibi = None
        if self.config.hybrid:
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


class LlamaHybridForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaHybridConfig):
        super().__init__(config)
        if config.alibi:
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = LlamaAttentionHybrid(config, i)
