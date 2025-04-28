# parse_xlam_dataset.py

from datasets import load_dataset, Dataset

def parse_xlam_huggingface():
    """
    Load the xLAM dataset directly from HuggingFace and parse it into prompt/completion pairs.
    """
    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    parsed_data = []

    for record in dataset:
        conversation = record.get("conversation", [])

        # Walk through conversation: pick user->assistant pairs
        for idx in range(len(conversation) - 1):
            current = conversation[idx]
            next_msg = conversation[idx + 1]

            if current["role"] == "user" and next_msg["role"] == "assistant":
                prompt = current["content"]
                completion = next_msg["content"]
                parsed_data.append({"prompt": prompt, "completion": completion})

    return parsed_data

def save_as_dataset(data_list, save_path=None):
    """
    Turn the parsed list into a HuggingFace Dataset.
    Optionally save if save_path provided.
    """
    dataset = Dataset.from_list(data_list)

    if save_path:
        dataset.save_to_disk(save_path)

    return dataset

if __name__ == "__main__":
    save_path = "xlam_data/parsed_dataset"  # or any path you want
    
    parsed = parse_xlam_huggingface()
    print(f"Parsed {len(parsed)} prompt/completion pairs.")

    dataset = save_as_dataset(parsed, save_path)
    print(f"Saved dataset to {save_path}.")
