from utils import ModelConfig, parse_config
from finetune import build_trainer, build_model


def finetune(config: ModelConfig):
    trainer = build_trainer(config)
    trainer.train()

def predict(config: ModelConfig):
    pass

def evaluate(config: ModelConfig):
    pass

def main():
    config = parse_config()
    
    if "train" in config["mode"]:
        finetune(config)
    if "test" in config["mode"]:
        predict(config)
        evaluate(config)

if __name__ == "__main__":
    main()