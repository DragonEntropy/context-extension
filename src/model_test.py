from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaConfig
import argparse
import torch

from alterations.FractionalRoPE import LlamaForCausalFractionalRoPE, LlamaConfigFractionalRoPE

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--model", type=str, default="models/llama-2-7b-hf")
    argparser.add_argument("-f", "--file_mode", action="store_true")
    args = argparser.parse_args()

    model_path = args.model

    base_config = LlamaConfig.from_pretrained(model_path)
    config = LlamaConfigFractionalRoPE(**base_config.to_dict(), fractional=True)
    model = LlamaForCausalFractionalRoPE.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).to(device)
    tokenizer.pad_token = tokenizer.eos_token

    if args.file_mode:
        path = input("Enter file path: ")
        while path:
            try:
                with open(path, 'r') as src:
                    prompt = src.read()
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    print(inputs["input_ids"].shape[1])
                    outputs = model.generate(**inputs, max_new_tokens=100)[0, inputs["input_ids"].shape[1]:]
                    true_output = tokenizer.decode(outputs, skip_special_tokens=True)

                    print(true_output)
            except FileNotFoundError:
                pass
            path = input("Enter file path: ")
    else:
        prompt = input("Enter your prompt: ")
        while prompt:
            line = input()
            while line:
                prompt = f"{prompt}\n{line}"
                line = input()
            print(f"Prompt length: {len(prompt.split(' '))}")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100)[0, inputs["input_ids"].shape[1]:]
            true_output = tokenizer.decode(outputs, skip_special_tokens=True)
            
            print(inputs["input_ids"].shape[1])
            print(true_output)
            prompt = input("Enter your prompt: ")

if __name__ == "__main__":
    main()