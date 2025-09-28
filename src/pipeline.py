from utils import ModelConfig, parse_config
from finetune import build_trainer, build_model
from pred import predict
from eval import evaluate


def finetune(config: ModelConfig):
    trainer = build_trainer(config)
    trainer.train()


def main():
    config = parse_config()
    if "train" in config["mode"]:
        finetune(config)
    if "test" in config["mode"]:
        predict(config)
    if "eval" in config["mode"]:
        evaluate(config)

if __name__ == "__main__":
    main()