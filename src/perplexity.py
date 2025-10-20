from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import math

from utils import build_model, ModelConfig
from tqdm import tqdm

# Evaluate library doesn't seem to have support for custom models, so using manual implementation
@torch.no_grad()
def compute_perplexity(config: ModelConfig):
    model_path = f"{config['save_dir']}/{config['model_name']}" if not config["eval_config"]["use_base_model"] else config["model_path"]
    tokeniser = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.model_max_length = config["new_context_length"]
    model = build_model(config, tokeniser, True).to("cuda").eval()

    ppl_offset = config["train_config"]["batch_size"] * config["train_config"]["train_steps"] + config["train_config"]["validation_size"]
    print(f"Starting index: {ppl_offset}", flush=True)
    dataset = load_dataset("common-pile/project_gutenberg_filtered", split="train", cache_dir="datasets")
    dataset = dataset.skip(ppl_offset)
    texts = dataset["text"][:config["eval_config"]["perplexity_examples"]]
    batch_size = 1 # config["train_config"]["batch_size"] // 2
    
    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        tokens = tokeniser(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["new_context_length"]
        ).to("cuda")

        
        outputs = model(**tokens, labels=tokens["input_ids"])
        loss = outputs.loss.item()

        # Counts non-padding tokens
        num_tokens = tokens["input_ids"].ne(tokeniser.pad_token_id).sum().item()

        # Weight the loss of each text extract by the number of tokens
        total_loss += loss * num_tokens
        total_tokens += num_tokens

    mean_loss = total_loss / total_tokens
    mean_perplexity = math.exp(mean_loss)


    print(f"Mean Perplexity: {mean_perplexity}")
    return mean_perplexity
