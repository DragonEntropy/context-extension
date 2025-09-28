import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import parse_config, ModelConfig
from finetune import build_model


dataset2prompt = json.load(open(f"config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open(f"config/dataset2maxlen.json", "r"))

"""
This code was adapted from the original LongBench source code:
    - https://aclanthology.org/2024.acl-long.172.pdf
    - https://github.com/THUDM/LongBench
"""
def build_chat(prompt, model_path):
    # This code is retained incase additional testing with llama2-chat was needed
    if "chat" in model_path:
        prompt = f"[INST]{prompt}[/INST]"


def get_pred(rank, data, max_length, max_gen, dataset, prompt_format, model, tokeniser, out_path, config: ModelConfig):
    device = torch.device(f'cuda:{rank}')
    max_gen = config[""]
    count = 0
    print(f"Dataset {dataset} has {length} entries", flush=True)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        length = json_obj["length"]

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokeniser(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokeniser.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokeniser.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(prompt, config["model_path"])
        input = tokeniser(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        # Retained for consistently even though this may not be an issue
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokeniser.eos_token_id, tokeniser.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokeniser.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

        count += 1

        # Ensures only first n entries are processed due to computational constraints
        if count == config["eval_config"]["max_per_dataset"]:
            break
    
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def predict(config):
    seed_everything(42)
    config = parse_config()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    tokeniser = AutoTokenizer.from_pretrained(config["base_path"])
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.model_max_length = config["new_context_length"]
    model = build_model(config, tokeniser)
    model_path = config["model_path"]
    model_name = f"{model_path}_{config["model_name"]}"
    
    if config["eval_config"]["long_bench_e"]:
        datasets = ["multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    # Iterate over every dataset
    for dataset in datasets:
        if config["eval_config"]["long_bench_e"]:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test', cache_dir="datasets", trust_remote_code=True)
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test', cache_dir="datasets", trust_remote_code=True)
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"

        # Only predict on english tasks for fairness
        if data[0]["language"] != "en":
            continue

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=get_pred,
                args=(rank, data_subsets[rank], config["new_context_length"], max_gen, prompt_format, dataset, model, tokeniser, out_path, config))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    config = parse_config()
    predict(config)
