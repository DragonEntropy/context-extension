from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model_path="../models/llama-2-7b-hf"
quant_path="../models/llama-2-custom"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)


# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model: Meta-Llama-3-8B-Instruct-FP8-Dynamic
model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)