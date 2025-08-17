## Setup

### 1. Install requirements.txt
```
pip install -r requirements.txt
```

### 2. Download benchmark
```
python
from datasets import load_dataset
dataset = load_dataset('THUDM/LongBench-v2', split='train')
```

### 2.5 Download git lfs if required
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt update
sudo apt install git-lfs

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
  --max-model-len 68000 \
  --gpu-memory-utilization 0.98
```

### 5. Start inference
```
python3 pred.py --model ../../models/llama-2-7b-hf
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