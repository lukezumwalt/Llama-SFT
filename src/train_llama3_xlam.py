# train_llama3_xlam.py

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # <-- Your model name here
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None:
        print("Loading model...")

        # Set up quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # safer type
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

        print("Model loaded successfully")

    return model, tokenizer


def prepare_lora(model):
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # typical for LLaMA3
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model

def preprocess_function(examples, tokenizer):
    """
    Merge prompt and completion into a single text and tokenize it.
    """
    inputs = []

    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        full_text = f"{prompt}\n\n###\n\n{completion}"
        inputs.append(full_text)

    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )

    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def main():
    dataset_path = "xlam_data/parsed_dataset"
    output_dir = "outputs/fine_tuned_llama3_xlam"

    # Load model and tokenizer
    model, tokenizer = load_model()
    model = prepare_lora(model)

    # Load and preprocess dataset
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True, remove_columns=dataset.column_names)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        report_to="none",
        bf16=True,  # Important for bfloat16 training
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        args=training_args,
    )

    trainer.train()

    # Save LoRA fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
