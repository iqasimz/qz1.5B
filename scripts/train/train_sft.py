import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk

# 1. Paths & model names
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "data/labels.jsonl"
OUTPUT_DIR = "checkpoints/sft_warmup"
CACHE_PATH = "cache/tokenized.arrow"

# 2. LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Load tokenizer & model with adapters
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = get_peft_model(model, lora_config)

# 4. Load and preprocess dataset (batched + caching)
def preprocess(batch):
    texts = []
    for example in batch["dialogue"]:
        convo = ""
        for turn in example:
            speaker = "User" if turn["speaker"] == "user" else "Assistant"
            convo += f"{speaker}: {turn['text']}\n"
        # Append the preferred response as the target
        # note: original shape of batch: list of dicts -> need to access response_preferred separately
        # but Hugging Face map with batched=True passes all columns as lists, so we process in a loop below
        texts.append(convo)
    # Now append responses
    tokenized = tokenizer(
        [t + f"Assistant: {resp}" for t, resp in zip(texts, batch["response_preferred"])],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    # For causal LM, labels = input_ids
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    return tokenized

# Load raw dataset
dataset = load_dataset("json", data_files={"train": DATA_PATH})

# Tokenize with caching
if os.path.isdir(CACHE_PATH):
    tokenized = load_from_disk(CACHE_PATH)
else:
    tokenized = dataset["train"].map(
        preprocess,
        batched=True,
        batch_size=32,
        remove_columns=dataset["train"].column_names,
    )
    tokenized.save_to_disk(CACHE_PATH)

# 5. Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    optim="adamw_torch",            # use PyTorch AdamW on MPS
    dataloader_num_workers=4,       # speed up data loading
    dataloader_pin_memory=True,     # further speed up
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    evaluation_strategy="no",       # change to "steps" if you add a validation split
    fp16=False,
    bf16=True,
    push_to_hub=False,
)

# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# 8. Run training
if __name__ == "__main__":
    trainer.train()

    # 9. Save LoRA adapters & tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapters"))
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ SFT warm-up complete. Adapters saved to", OUTPUT_DIR)