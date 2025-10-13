from utils import ModelConfig, parse_config
from finetune import build_trainer, build_model
from pred import predict
from eval import evaluate
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
    if "eval" in config["mode"]:
        evaluate(config)

if __name__ == "__main__":
    main()