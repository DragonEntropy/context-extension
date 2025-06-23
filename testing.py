from datasets import load_dataset
dataset = load_dataset('THUDM/LongBench-v2', split='train')

"""


"""
print(vars(dataset))
print(dataset[0].keys())
print(dataset[0]["choice_A"])