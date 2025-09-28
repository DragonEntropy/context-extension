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


class ModelConfig(TypedDict):
    train_config: TrainConfig
    eval_config: EvalConfig
    model_path: str
    model_type: Literal["base", "linear", "yarn", "fractional", "nope", "alibi"]
    old_context_length: int
    new_context_length: int
    mode: Literal["train", "test", "train-test"]


DEFAULT_TRAIN_CONFIG: TrainConfig = {
    "train_steps": 3200,
    "start_index": 0,
    "validation_size": 50,
    "batch_size": 10
}

DEFAULT_EVAL_CONFIG: EvalConfig = {
    "n_proc": 15,
    "max_per_dataset": -1
}

DEFAULT_MODEL_CONFIG: ModelConfig = {
    "train_config": DEFAULT_TRAIN_CONFIG,
    "eval_config": DEFAULT_EVAL_CONFIG,
    "model_path": "models/llama-2-7b-hf",
    "model_type": "base",
    "old_context_length": 4096,
    "new_context_length": 8192,
    "mode": "train-test"
}

def parse_config():
    argparser = ArgumentParser()
    argparser.add_argument("-c", "--config_path", type=str, default="pipeline_config.json", help="Specifies the default config file for the pipeline")
    args = argparser.parse_args()
    try:
        with open(args.config_path, "r") as config_file:
            config = json.load(config_file)
            # Merge loaded configs with defaults
            config["train_config"] = cast(TrainConfig, {**DEFAULT_TRAIN_CONFIG, **config["train_config"]})
            config["eval_config"] = cast(EvalConfig, {**DEFAULT_EVAL_CONFIG, **config["eval_config"]})
            return cast(ModelConfig, {**DEFAULT_MODEL_CONFIG, **config})
    except FileNotFoundError:
        assert False, f"Pipeline config file '{args.config_path}' was not found. Please verify that the config path specified by the -c flag is correct."
    return None