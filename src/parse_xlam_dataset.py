# parse_xlam_dataset.py

import json
from datasets import load_dataset, Dataset

def parse_xlam_huggingface():
    """
    Load the xLAM tool-calling dataset and parse it into prompt/completion pairs.
    """
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    parsed_data = []

    for record in dataset:
        query = record.get("query", "").strip()
        answers_json_str = record.get("answers", "").strip()

        if not query or not answers_json_str:
            continue

        try:
            answers = json.loads(answers_json_str)
        except json.JSONDecodeError:
            continue  # skip bad record

        # Format the answers nicely (you could adjust formatting depending on your use case)
        completion = json.dumps(answers, indent=2)

        parsed_data.append({
            "prompt": query,
            "completion": completion
        })

    return parsed_data


def save_as_dataset(data_list, save_path=None):
    """
    Turn the parsed list into a HuggingFace Dataset.
    Optionally save if save_path provided.
    """
    if len(data_list) == 0:
        raise ValueError("No valid examples found. Dataset is empty!")

    dataset = Dataset.from_list(data_list)

    if save_path:
        dataset.save_to_disk(save_path)

    return dataset

if __name__ == "__main__":
    save_path = "xlam_data/parsed_dataset"

    parsed = parse_xlam_huggingface()
    print(f"Parsed {len(parsed)} prompt/completion pairs.")

    dataset = save_as_dataset(parsed, save_path)
    print(f"Saved dataset to {save_path}.")
