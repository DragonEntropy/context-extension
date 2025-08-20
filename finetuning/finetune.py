from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TrainerCallback
import math
import torch

model_path = "../../models/llama-2-7b-hf"


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.min_loss = math.inf
        self.counter = 0
        self.eval_step = 0

    def on_step_end(self, args, state, control, metrics, **kwargs):
        score = metrics.get("eval_loss", math.inf)
        print(f"Eval step {self.eval_step} - Current eval loss: {score}")

        if score < self.min_loss:
            self.min_loss = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at step {self.eval_step} with eval loss: {score}")
                control.should_training_stop = True
        eval_step += 1


def tokenise(batch, tokeniser):
    tokens = tokeniser(
        batch["text"],
        truncation=True,
        max_length=tokeniser.model_max_length,
        padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    tokeniser = AutoTokenizer.from_pretrained(model_path)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.model_max_length = 4096

    # Dataset is being streamed due to storage constraints
    dataset = load_dataset("common-pile/wikimedia_filtered", split="train", streaming=True)
    fields = list(next(iter(dataset)).keys())
    dataset = dataset.map(lambda data: tokenise(data, tokeniser), remove_columns=fields)
    eval_dataset = dataset.take(1000)
    train_dataset = dataset.skip(1000)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser,
        mlm=False,
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.config.pad_token_id = tokeniser.eos_token_id

    training_args = TrainingArguments(
        output_dir=f"{model_path}_finetuned",
        eval_strategy="steps",
        save_strategy="steps",  
        eval_steps=10,
        save_steps=10,
        save_total_limit=2,
        logging_steps=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps = 1000,
        remove_unused_columns=False,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokeniser,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=3)]
    )

    trainer.train()


if __name__ == "__main__":
    main()
