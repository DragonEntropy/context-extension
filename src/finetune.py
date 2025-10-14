from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback
)
from bitsandbytes.optim import PagedAdamW8bit
import math

from utils import parse_config, ModelConfig, build_model


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=10):
        self.patience = patience
        self.min_loss = math.inf
        self.counter = 0
        self.eval_step = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        score = metrics.get("eval_loss", math.inf)
        print(f"Eval step {self.eval_step} - Current eval loss: {score}", flush=True)

        if score < self.min_loss:
            self.min_loss = score
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at step {self.eval_step} with eval loss: {score}", flush=True)
                control.should_training_stop = True
        self.eval_step += 1
        return control


def tokenise(batch, tokeniser):
    tokens = tokeniser(
        batch["text"],
        truncation=True,
        max_length=tokeniser.model_max_length,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def build_trainer(config: ModelConfig):
    # Tokeniser setup
    tokeniser = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.model_max_length = config["new_context_length"]

    # Dataset partitioning
    dataset = load_dataset("common-pile/project_gutenberg_filtered", split="train", cache_dir="datasets")
    # Only need the tokenised text
    fields = list(next(iter(dataset)).keys())
    dataset = dataset.map(
        lambda data: tokenise(data, tokeniser),
        remove_columns=fields, 
        load_from_cache_file=True,
        cache_file_name="datasets/cached_pg19"
    )

    validation_size = config["train_config"]["validation_size"]
    eval_dataset = dataset.take(validation_size)
    train_dataset = dataset.skip(validation_size + config["train_config"]["start_index"])

    # Finetune only on next token prediction like original Llama2
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser,
        mlm=False,
    )

    # Model setup
    model = build_model(config, tokeniser, False)

    # Optimiser setup
    optimiser = PagedAdamW8bit(model.parameters(), lr=1e-4)
    scheduler = None

    # Trainer setup
    output_model_path = f"{config['save_dir']}/{config['model_name']}"
    training_args = TrainingArguments(
        output_dir=output_model_path,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        save_total_limit=1,
        logging_steps=100,
        per_device_train_batch_size=config["train_config"]["batch_size"],
        per_device_eval_batch_size=config["train_config"]["batch_size"] // 2,
        max_steps=config["train_config"]["train_steps"],
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokeniser,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=5)],
        optimizers=(optimiser, scheduler)
    )

    tokeniser.save_pretrained(f"{config['save_dir']}/{config['model_name']}")
    return trainer


if __name__ == "__main__":
    config = parse_config()
    trainer = build_trainer(config)
    trainer.train()
