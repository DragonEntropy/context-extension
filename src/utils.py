from typing import TypedDict, Literal, cast
from argparse import ArgumentParser
import json


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
    model_type: Literal["base", "linear", "yarn", "fractional", "nope", "alibi"]
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