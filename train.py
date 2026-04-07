import torch
import time
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

def main():
    # Defining the base model (Using GPT-2 as a lightweight test architecture)
    model_name = "gpt2"
    
    print(f"Loading Tokenizer and Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Downloading and Preparing Dataset...")
    # Load a sample of the Finance Alpaca dataset
    dataset = load_dataset("gbharti/finance-alpaca", split="train[:1000]") 

    # Formating the instruction, input, and output into a single text block
    def format_data(example):
        return {"text": f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: {example['output']}"}

    # Tokenizing the text to convert strings into model-readable tensors
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    # Applying  transformations
    formatted_dataset = dataset.map(format_data)
    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
    
    # Cleaning up dataset columns for PyTorch compatibility
    tokenized_dataset = tokenized_dataset.remove_columns(['instruction', 'input', 'output', 'text'])
    tokenized_dataset.set_format("torch")
    
    # Dynamically generating labels for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Initializing DeepSpeed Trainer")
    # Define training arguments with DeepSpeed and Mixed Precision enabled
    training_args = TrainingArguments(
        output_dir="./results_deepspeed",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        fp16=True,                          # Enable Mixed Precision (16-bit)
        deepspeed="ds_config.json",         # Inject DeepSpeed ZeRO Configuration
        logging_steps=50,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Executing DeepSpeed Training Pipeline")
    start_time = time.time()
    
    # Start the training loop
    trainer.train()
    
    print(f"\n Training Complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
