# train_llama3_xlam.py

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
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

def main():
    model_path = "path/to/your/llama3.2-3b-instruct"
    dataset_path = "xlam_data/parsed_dataset"
    output_dir = "outputs/fine_tuned_llama3_xlam"

    # Load
    model, tokenizer = load_model(model_path)
    model = prepare_lora(model)

    dataset = load_from_disk(dataset_path)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
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
        bf16=True,  # If your GPU supports bf16
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        args=training_args,
    )

    trainer.train()

    # Save LoRA
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()
