from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback
)
import math
from bitsandbytes.optim import AdamW8bit
from argparse import ArgumentParser

model_length = 2**(18 - 4)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=1):
        self.patience = patience
        self.min_loss = math.inf
        self.counter = 0
        self.eval_step = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        score = metrics.get("eval_loss", math.inf)
        print(f"Eval step {self.eval_step} - Current eval loss: {score}", flush=True)

        if score < self.min_loss:
            self.min_loss = score
            self.counter = 0
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
        padding=False
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    argparser = ArgumentParser()
    argparser.add_argument("-m", "--model_path", type=str, default="models/llama-2-7b-hf")
    argparser.add_argument("-s", "--start_index", type=int, default=0)
    args = argparser.parse_args()
    input_model_path = args.model_path
    output_model_path = f"{args.model_path}_finetuned"
    start_index = args.start_index

    tokeniser = AutoTokenizer.from_pretrained(input_model_path)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.model_max_length = model_length

    # Dataset is being streamed due to storage constraints
    dataset = load_dataset("common-pile/project_gutenberg_filtered", split="train", streaming=True)
    fields = list(next(iter(dataset)).keys())
    dataset = dataset.map(lambda data: tokenise(data, tokeniser), remove_columns=fields)
    eval_dataset = dataset.take(25)
    train_dataset = dataset.skip(25 + start_index)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser,
        mlm=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        input_model_path,
        device_map="auto"
    )
    model.config.pad_token_id = tokeniser.eos_token_id
    model.gradient_checkpointing_enable()

    # Adam 8 bit
    optimiser = AdamW8bit(model.parameters(), lr=1e-4, optim_bits=32)
    scheduler = None

    training_args = TrainingArguments(
        output_dir=output_model_path,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        save_total_limit=1,
        logging_steps=100,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=1000 - start_index,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokeniser,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=1)],
        optimizers=(optimiser, scheduler)
    )

    trainer.train()


if __name__ == "__main__":
    main()
