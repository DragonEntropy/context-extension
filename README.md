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

### Pipeline code (recommended over running individual components)
Edit the json file at default file path ```config/pipeline_config.json``` to control outputs.
See ```src/utils.py``` for json file format.
Alternatively, can control the json file path with the -c flag
```
nohup python3 src/pipeline.py > output.log 2>&1 &
```

### Finetuning
```
nohup python3 src/finetune.py > logs/finetune_output.log 2>&1 &
```

### Prediction (LongBench)
```
nohup python3 src/pred.py > logs/pred_output.log 2>&1 &
```

### Evaluation (perplexity)
```
python3 src/eval.py > logs/eval_output.log 2>&1 &
```


## Other commands
### Launch model with vllm (for LongBenchv2):
```
vllm serve models/llama-2-7b-hf \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.98 \
  --quantization bitsandbytes
```
