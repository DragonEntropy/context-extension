## Setup

### 1. Install requirements.txt
```
pip install -r requirements.txt
```

### 2. Download benchmark
```
python
from datasets import load_dataset
dataset = load_dataset('THUDM/LongBench-v2', split='train', cache_dir='datasets')

# Requires downgrading to datasets 2.19.1 or earlier 
datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
  "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
  "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test', cache_dir='datasets')
```

### 2.5 Download git lfs if required
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt update
apt install git-lfs

### 3. Llama to huggingface
```
python context-extension/torch_to_hf.py \
  --input_dir models/Llama-2-7b/ \
  --model_size 7B \
  --output_dir models/llama-2-7b-hf
```

### 4. Start model
In the LongBench folder:
```
vllm serve ../../models/llama-2-7b-hf \
  --max-model-len 68200 \
  --gpu-memory-utilization 0.98 \
  --quantization bitsandbytes
```

### 5. Start inference
```
python3 pred.py --model ../../models/llama-2-7b-hf
```

### 6. Training (in background)
```
nohup python finetune.py > output.log 2>&1 &
```

### 7. awq quantisation (for vllm compatibility)
python3 -m awq.quantize \
  --model_path ../../models/llama-2-7b-hf \
  --quant_path ../../models/llama-2-custom \
  --w_bit 8 \
  --q_group_size 128 \
  --use_sym


## Info

Longbench data entries follow the format:
```
json
{
    "_id": "Unique identifier for each piece of data",
    "domain": "The primary domain category of the data",
    "sub_domain": "The specific sub-domain category within the domain",
    "difficulty": "The difficulty level of the task, either 'easy' or 'hard'",
    "length": "The length category of the task, which can be 'short', 'medium', or 'long'",
    "question": "The input/command for the task, usually short, such as questions in QA, queries in many-shot learning, etc",
    "choice_A": "Option A", "choice_B": "Option B", "choice_C": "Option C", "choice_D": "Option D",
    "answer": "The groundtruth answer, denoted as A, B, C, or D",
    "context": "The long context required for the task, such as documents, books, code repositories, etc."
}
```