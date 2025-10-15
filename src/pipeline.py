from utils import ModelConfig, parse_config
from finetune import build_trainer
from pred import predict
from eval import evaluate
from perplexity import compute_perplexity

import torch
import gc


def finetune(config: ModelConfig):
    trainer = build_trainer(config)
    trainer.train()
    trainer.save_model(f"{config['save_dir']}/{config['model_name']}")

    del trainer.model
    trainer.model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def main():
    config = parse_config()
    print(f"Pipeline config: {config}")
    if "train" in config["mode"]:
        finetune(config)
    if "test" in config["mode"]:
        predict(config)
        evaluate(config)
    if "eval" in config["mode"]:
        if config["eval_config"]["perplexity_examples"]:
            compute_perplexity()


if __name__ == "__main__":
    main()
