import torch
import math
from typing import Callable, Optional

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaRotaryEmbedding
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from transformers.utils.generic import TransformersKwargs



class LlamaFractionalRoPEConfig(LlamaConfig):
    def __init__(self, fractional=True, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.fractional = fractional
        self.alpha = alpha


class LlamaFractionalRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: LlamaConfig, device=None, alpha=1, L=16384, l=2048):
        # Fractional RoPE works with any variant of RoPE that doesn't mess with position ids.
        super().__init__(config, device=device)
        self.l = l
        self.L = L
        self.alpha = alpha
        self.beta = math.pow(l, -alpha) - math.pow(L, -alpha)

        # Override inv_freq
        self.inv_freq = None
        self.precompute_angles(device)

    def precompute_angles(self, device):
        dim = self.config.head_dim
        base = self.config.rope_theta
        max_len = self.config.max_position_embeddings

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)).unsqueeze(0)
        rel_pos_ids = torch.arange(-max_len + 1, max_len).to(device=device, dtype=torch.float)
        scaled_pos_ids = self.fractional_function(rel_pos_ids)

        angles = scaled_pos_ids.unsqueeze(1) @ inv_freq.unsqueeze(0)
        emb = torch.cat((angles, angles), dim=-1)
        cos_cache = emb.cos()
        sin_cache = emb.sin()
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)

    def fractional_function(self, x):
        # return torch.clamp(x, max=self.l)
        if self.alpha == 0:
            return x * self.l / self.L
        return x / torch.pow(1 + self.beta * torch.pow(x, self.alpha), 1 / self.alpha)

    @torch.no_grad()
    def forward(self, x, position_ids):

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            max_len = self.config.max_position_embeddings

            if position_ids.dim() > 1:
                position_ids = position_ids[0]

            # Position id 0 is at max_len - 1 in the cache
            indices = (position_ids +  max_len - 1).clamp(0, 2 * max_len - 2)

            cos = self.cos_cache[:, indices, :]
            sin = self.sin_cache[:, indices, :]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Adapted from transformers LlamaAttention
class LlamaAttentionFractionalRoPE(LlamaAttention):
    def __init__(self, config: LlamaFractionalRoPEConfig, layer_idx: int):
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


class LlamaFractionalRoPEForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaFractionalRoPEConfig):
        super().__init__(config)
        if config.fractional:
            self.model.rotary_emb = LlamaFractionalRotaryEmbedding(config=config, alpha=0, L=16384, l=4096)
