from evaluate import load
from transformers import AutoTokenizer
from datasets import load_dataset

from utils import build_model, ModelConfig


def compute_perplexity(config: ModelConfig):
    model_path = f"{config['save_dir']}/{config['model_name']}" if not config["eval_config"]["use_base_model"] else config["model_path"]
    tokeniser = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = build_model(config, tokeniser, True)

    ppl_offset = config["train_config"]["batch_size"] * config["train_config"]["train_steps"] + config["train_config"]["validation_size"]
    dataset = load_dataset("common-pile/project_gutenberg_filtered", split="train", cache_dir="datasets")
    dataset = dataset.skip(ppl_offset)
    ppl = load("perplexity")

    results = ppl.compute(
        model=model,
        tokenizer=tokeniser,
        input_texts=dataset["text"][:config["eval_config"]["perplexity_examples"]],
        device="cuda",
        batch_size=config["train_config"]["batch_size"] // 2,
        max_length=config["new_context_length"]
    )

    print(f"Mean perplexity: {results["mean_perplexity"]}\nPerplexities: {results["perplexities"]}")
