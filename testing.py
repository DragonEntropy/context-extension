from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

print(torch.cuda.is_available())

dataset = load_dataset('THUDM/LongBench-v2', split='train', streaming=True)

# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
tokenizer = LlamaTokenizer.from_pretrained("../models/llama-2-7b-hf", local_files_only=True)
model = LlamaForCausalLM.from_pretrained(
    "../models/llama-2-7b-hf",
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto"
).half().to("cuda")

def test_example(tokeniser, model, example):
    inputs = tokeniser(example["question"], return_tensors="pt").to("cuda")
    outputs = tokenizer.decode(model.generate(**inputs, max_new_tokens=100)[0], skip_special_tokens=True)
    print(f"\n\nQuestion:\n{example['question']}")
    print(f"\nAnswer:\n{outputs}")

print(vars(dataset))
count = 0
for example in dataset:
    test_example(tokenizer, model, example)
    count += 1
    if count >= 5:
        break
