from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, LlamaConfig

from alterations.ALiBi import LlamaAttentionALiBi
from alterations.NoPE import LlamaAttentionNoPE


class LlamaAlternatingConfig(LlamaConfig):
    def __init__(self, alt_pattern=["nope", "rope"], **kwargs):
        super().__init__(**kwargs)
        self.alt_pattern = alt_pattern


class LlamaAlternatingForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaAlternatingConfig):
        super().__init__(config)
        if config.alt_pattern is not None:
            for i, layer in enumerate(self.model.layers):
                attention_type = config.alt_pattern[i % len(config.alt_pattern)]

                if attention_type == "alibi":
                    print(f"Configuring layer {i} to ALiBi")
                    old_attn = layer.self_attn
                    layer.self_attn = LlamaAttentionALiBi(config, i)
                    layer.self_attn.load_state_dict(old_attn.state_dict())
                elif attention_type == "nope":
                    print(f"Configuring layer {i} to NoPE")
                    layer.self_attn = LlamaAttentionNoPE(config, i)
                elif attention_type == "rope":
                    print(f"Configuring layer {i} to RoPE")
                    layer.self_attn = LlamaAttention(config, i)
                else:
                    assert False, f"Unimplemented attention type {attention_type}."
