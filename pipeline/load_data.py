
from datasets import load_dataset
import json, os

dataset = load_dataset("google-research-datasets/mbpp", split="test")
filtered_dataset = []
#extract only essential data from mbpp dataset
for i in dataset:
    filtered_dataset.append({
        "task_id":i["task_id"],
        "text":i["text"],
        "test_list":i["test_list"]
    })

with open("filtered_dataset.json", "w") as f:
    json.dump(filtered_dataset, f, indent=2)
    