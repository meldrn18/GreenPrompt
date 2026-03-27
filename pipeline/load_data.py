
from datasets import load_dataset
import json, os

def load_data():
    dataset = load_dataset("google-research-datasets/mbpp", split="test")
    filtered_dataset = []
    #extract only essential data from mbpp dataset
    for i in dataset:
        filtered_dataset.append({
            "task_id":i["task_id"],
            "text":i["text"],
            "solution":i["code"],
            "test_list":i["test_list"]
        })

    with open("data/filtered_dataset.json", "w") as f:
        json.dump(filtered_dataset, f, indent=2)
    print(f"Total problems: {len(filtered_dataset)}")

def create_examples(input_path="data/filtered_dataset.json", output_path="data/mbpp_examples.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #load filtered MBPP dataset
    with open(input_path, "r") as f:
        dataset = json.load(f)

    #build dictionary keyed by task_id
    examples = {}
    for entry in dataset:
        task_id = str(entry["task_id"])
        examples[task_id] = {
            "text": entry["text"],
            "solution": entry["solution"]
        }

    #save to JSON file
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=4)

    print(f"Few-shot examples saved to {output_path}")
    print(f"Total examples: {len(examples)}")

def merge_examples(filtered_path = "data/filtered_dataset.json",
                        examples_path="data/mbpp_examples.json",
                        output_path="data/merged_dataset.json"):
    with open(filtered_path, "r") as f:
        filtered_dataset = json.load(f)
    with open(examples_path, "r") as f:
        examples = json.load(f)
        if isinstance(examples, dict):
            examples_dict = {int(k): v for k, v in examples.items()}
        else:
            examples_dict = {ex["task_id"]: ex["examples"] for ex in examples}
    for item in filtered_dataset:
        tid = item["task_id"]
        item["examples"] = examples_dict.get(tid,[])
    with open(output_path, "w") as f:
        json.dump(filtered_dataset, f, indent=2)
    print(f"Total merged tasks: {len(filtered_dataset)}")


if __name__ == "__main__":
    load_data()
    merge_examples()
    