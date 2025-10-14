from typing import TypedDict, Literal, cast
from argparse import ArgumentParser
import json
import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from alterations.FractionalRoPE import LlamaFractionalRoPEForCausalLM, LlamaFractionalRoPEConfig
from alterations.NoPE import LlamaNoPEForCausalLM, LlamaNoPEConfig
from alterations.ALiBi import LlamaALiBiForCausalLM, LlamaALiBiConfig
from alterations.Hybrid import LlamaHybridForCausalLM, LlamaHybridConfig


class TrainConfig(TypedDict):
    train_steps: int
    start_index: int
    validation_size: int
    batch_size: int


class EvalConfig(TypedDict):
    n_proc: int
    max_per_dataset: int
    long_bench_e: bool
    use_base_model: bool


class ModelConfig(TypedDict):
    train_config: TrainConfig
    eval_config: EvalConfig
    model_path: str
    save_dir: str
    model_name: str
    model_type: Literal["base", "linear", "yarn", "fractional", "nope", "alibi", "hybrid"]
    old_context_length: int
    new_context_length: int
    mode: Literal["train", "test", "eval", "train-test", "test-eval", "train-test-eval"]


DEFAULT_TRAIN_CONFIG: TrainConfig = {
    "train_steps": 3200,
    "start_index": 0,
    "validation_size": 50,
    "batch_size": 10
}

DEFAULT_EVAL_CONFIG: EvalConfig = {
    "n_proc": 15,
    "max_per_dataset": -1,
    "long_bench_e": False,
    "use_base_model": False
}

DEFAULT_MODEL_CONFIG: ModelConfig = {
    "train_config": DEFAULT_TRAIN_CONFIG,
    "eval_config": DEFAULT_EVAL_CONFIG,
    "model_path": "models/llama2-7b-hf",
    "save_dir": "models",
    "model_name": "test_model",
    "model_type": "base",
    "old_context_length": 4096,
    "new_context_length": 8192,
    "mode": "train-test-eval"
}

def parse_config():
    argparser = ArgumentParser()
    argparser.add_argument("-c", "--config_path", type=str, default="config/pipeline_config.json", help="Specifies the default config file for the pipeline")
    args = argparser.parse_args()
    try:
        with open(args.config_path, "r") as config_file:
            config = json.load(config_file)
            # Merge loaded configs with defaults
            if "train_config" in config.keys():
                config["train_config"] = cast(TrainConfig, {**DEFAULT_TRAIN_CONFIG, **config["train_config"]})
            else:
                config["train_config"] = DEFAULT_TRAIN_CONFIG
            if "eval_config" in config.keys():
                config["eval_config"] = cast(EvalConfig, {**DEFAULT_EVAL_CONFIG, **config["eval_config"]})
            else:
                config["eval_config"] = DEFAULT_EVAL_CONFIG
            return cast(ModelConfig, {**DEFAULT_MODEL_CONFIG, **config})
    except FileNotFoundError:
        assert False, f"Pipeline config file '{args.config_path}' was not found. Please verify that the config path specified by the -c flag is correct."
    return None


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
    elif model_type == "hybrid":
        config = LlamaHybridConfig(**base_config.to_dict())
        config.architectures = ["LlamaHybridForCausalLM"]
        model = LlamaHybridForCausalLM.from_pretrained(
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