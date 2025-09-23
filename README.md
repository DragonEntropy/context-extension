## Setup
Repository should be structured as follows:
```
.
├── datasets/
├── logs/
├── src/
│   ├── alterations/
|   │   └── <model_code>.py
│   ├── LongBench/ (subrepo)
│   └── <src_code>.py
├── .gitignore
├── .gitmodules
├── README.md
└── requirements.txt
```

### 1. Install requirements.txt (recommended in a python virtual environment)
```
pip install -r requirements.txt
```

### 1.5 Download git lfs if required for model download
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt update
apt install git-lfs
```

### 2. Download llama2-7b model into models folder
Accessible via:
- Huggingface: https://huggingface.co/meta-llama/Llama-2-7b
- META website: https://www.llama.com/llama-downloads

### 3. Convert llama2 to transformers model
```
python src/torch_to_hf.py \
  --input_dir models/Llama-2-7b/ \
  --model_size 7B \
  --output_dir models/llama-2-7b-hf
```
The original Llama2 model can be deleted at this point

## Model running

### Finetuning
nohup python3 src/finetune.py > logs/finetune_output.log 2>&1 &

### Evaluating
nohup python3 src/LongBench/LongBench/pred.py --model llama2-7b -l 40 > logs/eval_output.log 2>&1 &

## Other commands
### Launch model with vllm (for LongBenchv2):
```
vllm serve models/llama-2-7b-hf \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.98 \
  --quantization bitsandbytes
```

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