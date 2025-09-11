import torch
from typing import Callable, Optional

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaRotaryEmbedding
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from transformers.utils.generic import TransformersKwargs


class LlamaConfigFractionalRoPE(LlamaConfig):
    def __init__(self, fractional=True, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.fractional = fractional
        self.alpha = alpha


class LlamaFractionalRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: LlamaConfig, device=None, alpha=1, L=2048, l=2048):
        # Fractional RoPE works with any variant of RoPE that doesn't mess with position ids.
        super().__init__(config, device=device)
        self.alpha = alpha
        self.beta = torch.pow(l, -alpha) - torch.pow(L, -alpha)

    def fractional_function(self, x):
        return x / torch.pow(1 + self.beta * torch.pow(x, self.alpha), 1 / self.alpha)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

        # Key change: Rescale the position ids
        position_ids_expanded = self.fractional_function(position_ids[:, None, :].float())

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Adapted from transformers LlamaAttention
class LlamaAttentionFractionalRoPE(LlamaAttention):
    def __init__(self, config: LlamaConfigFractionalRoPE, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Main modification: Redefine the application of RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaForCausalFractionalRoPE(LlamaForCausalLM):
    def __init__(self, config: LlamaConfigFractionalRoPE):
        super().__init__(config)
        if config.fractional:
            self.model.rotary_emb = LlamaFractionalRotaryEmbedding(config=config, alpha=1, L=2048, l=2048)
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = LlamaAttentionFractionalRoPE(config, i)
