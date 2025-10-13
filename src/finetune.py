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
from utils import parse_config, ModelConfig


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


def build_model(json_config: ModelConfig, tokeniser, is_eval: bool):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extension_ratio = float(json_config["new_context_length"] / json_config["old_context_length"])
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
            "factor": extension_ratio,
            "original_max_position_embeddings": json_config["old_context_length"]
        }
    }

    model_type = json_config["model_type"]
    if is_eval and not json_config["eval_config"]["use_base_model"]:
        model_path = f"{json_config['save_dir']}/{json_config['model_name']}" 
    else:
        model_path = json_config["model_path"]
    base_config = LlamaConfig.from_pretrained(model_path)
    print(f"Attempting to run model {json_config['model_name']} of type {model_type}", flush=True)
    if model_type in default_model_rope_config.keys():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=base_config,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to(device)
        model.config.rope_scaling = default_model_rope_config[model_type]

    elif model_type == "fractional":
        config = LlamaFractionalRoPEConfig(**base_config.to_dict())
        config.architectures = ["LlamaFractionalRoPEForCausalLM"]
        model = LlamaFractionalRoPEForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to(device)
    elif model_type == "nope":
        config = LlamaNoPEConfig(**base_config.to_dict())
        config.architectures = ["LlamaNoPEForCausalLM"]
        model = LlamaNoPEForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to(device)
    elif model_type == "alibi":
        config = LlamaALiBiConfig(**base_config.to_dict())
        config.architectures = ["LlamaALiBiForCausalLM"]
        model = LlamaALiBiForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to(device)
    else:
        assert False, "Model type not supported. Model type can be customised by changing the model_type attribute in config."

    model.config.pad_token_id = tokeniser.eos_token_id
    model.config.max_position_embeddings = json_config["new_context_length"]
    model.gradient_checkpointing_enable()
    print(f"Successfully loaded model type {type(model).__name__}")
    print(f"Model config:\n{model.config}")
    return model


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
