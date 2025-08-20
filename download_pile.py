from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("common-pile/wikimedia_filtered", cache_dir="../datasets")