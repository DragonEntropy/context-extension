from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--model", type=str, default="models/llama-2-7b-hf")
    args = argparser.parse_args()

    model_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    prompt = input("Enter your prompt: ")
    while prompt:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        prompt = input("Enter your prompt: ")

if __name__ == "__main__":
    main()