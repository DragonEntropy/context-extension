from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback
)
from transformers.models.llama.configuration_llama import LlamaConfig
from bitsandbytes.optim import PagedAdamW8bit
from argparse import ArgumentParser
import math
import torch

from alterations.FractionalRoPE import LlamaFractionalRoPEForCausalLM, LlamaFractionalRoPEConfig
from alterations.NoPE import LlamaNoPEForCausalLM, LlamaNoPEConfig
from alterations.ALiBi import LlamaALiBiForCausalLM, LlamaALiBiConfig


BASE_CONTEXT_LENGTH = 4096

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


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("-t", "--model_type", type=str, default="base", help="Specifies the input model type")
    argparser.add_argument("-p", "--model_path", type=str, default="models/llama-2-7b-hf", help="Specifies the input model path")
    argparser.add_argument("-i", "--iterations", type=int, default=3200, help="Specifies the maximum number of training iterations")
    argparser.add_argument("-s", "--start_index", type=int, default=0, help="Specifies the start index in the train dataset (useful for resuming finetuning)")
    argparser.add_argument("-v", "--validation_size", type=int, default=50, help="Specifies the size of the validation dataset. Should be consistent for fair comparison.")
    argparser.add_argument("-b", "--batch_size", type=int, default=10, help="Specifies the training and evaluation batch sizes.")
    argparser.add_argument("-c", "--context_size", type=int, default=8192, help="Specifies the size of the context window.")

    return argparser.parse_args()


def tokenise(batch, tokeniser):
    tokens = tokeniser(
        batch["text"],
        truncation=True,
        max_length=tokeniser.model_max_length,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def build_model(args, tokeniser):
    """
    List of supported RoPE configurations:
        "base": No interpolation
        "linear": Linear interpolation
        "dynamic": Dynamic ntk-aware RoPE interpolation
        "yarn": Yet another RoPE extension
        "fractional:" Custom non-linear relative position implementation
        "nope": No positional embeddings
        "alibi": Attention with linear biases
    """

    extension_ratio = float(args.context_size / BASE_CONTEXT_LENGTH)
    default_model_rope_config = {
        "base": None,
        "linear": {
            "rope_type": "linear",
            "factor": extension_ratio
        },
        "dynamic": {
            "rope_type": "dynamic",
            "factor": extension_ratio
        },
        "yarn": {
            "rope_type": "yarn",
            "factor": extension_ratio
        }
    }

    base_config = LlamaConfig.from_pretrained(args.model_path)
    print(f"Attempting to run model {args.model_type}", flush=True)
    if args.model_type in default_model_rope_config.keys():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=base_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.config.rope_scaling = default_model_rope_config[args.model_type]

    elif args.model_type == "fractional":
        config = LlamaFractionalRoPEConfig(**base_config.to_dict(), fractional=True, alpha=1)
        model = LlamaFractionalRoPEForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif args.model_type == "nope":
        config = LlamaNoPEConfig(**base_config.to_dict(), nope=True)
        model = LlamaNoPEForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif args.model_type == "alibi":
        config = LlamaALiBiConfig(**base_config.to_dict(), alibi=True)
        model = LlamaALiBiForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        assert False, "Model type not supported. Model type can be customised by using the -t or --model_type flag."

    model.config.pad_token_id = tokeniser.eos_token_id
    model.config.max_position_embeddings = args.context_size
    model.gradient_checkpointing_enable()
    print(model.config, flush=True)
    return model


def build_trainer():
    args = parse_args()

    # Tokeniser setup
    tokeniser = AutoTokenizer.from_pretrained(args.model_path)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.model_max_length = args.context_size

    # Dataset partitioning
    dataset = load_dataset("common-pile/project_gutenberg_filtered", split="train", streaming=True)
    # Only need the tokenised text
    fields = list(next(iter(dataset)).keys())
    dataset = dataset.map(lambda data: tokenise(data, tokeniser), remove_columns=fields)
    eval_dataset = dataset.take(args.validation_size)
    train_dataset = dataset.skip(args.validation_size + args.start_index)

    # Finetune only on next token prediction like original Llama2
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokeniser,
        mlm=False,
    )

    # Model setup
    model = build_model(args, tokeniser)

    # Optimiser setup
    optimiser = PagedAdamW8bit(model.parameters(), lr=1e-4)
    scheduler = None

    # Trainer setup
    output_model_path = f"{args.model_path}_finetuned"
    training_args = TrainingArguments(
        output_dir=output_model_path,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100,
        save_total_limit=1,
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size // 2,
        max_steps=args.iterations,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokeniser,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(patience=3)],
        optimizers=(optimiser, scheduler)
    )

    return trainer

def main():
    trainer = build_trainer()
    trainer.train()
    

if __name__ == "__main__":
    main()
