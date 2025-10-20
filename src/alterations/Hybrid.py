import torch
from typing import Callable, Optional

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, LlamaAttention

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
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Using original RoPE definition
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Cache offset need to correctly compute alibi
        cache_offset = 0
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # past_key_values is a DynamicCache object, a subclass of the Cache object
            cache_offset = past_key_value.get_seq_length(self.layer_idx) - query_states.shape[-2]
        
        query_len = query_states.shape[-2]
        key_len = key_states.shape[-2]

        alibi = None
        attention_interface: Callable = eager_attention_forward
        if self.config.hybrid:
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


class LlamaHybridForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaHybridConfig):
        super().__init__(config)
        if config.hybrid:
            for i, layer in enumerate(self.model.layers):
                # Need to also copy attention weights
                old_attn = layer.self_attn
                layer.self_attn = LlamaAttentionHybrid(config, i)
                layer.self_attn.load_state_dict(old_attn.state_dict())
