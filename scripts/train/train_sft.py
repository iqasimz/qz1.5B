import os
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk

# 1. Paths & model names
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "data/newlabels.jsonl"
OUTPUT_DIR = "checkpoints/sft"
CACHE_PATH = "cache1/tokenized.arrow"

# 2. Force CPU usage and disable MPS
print("Forcing CPU usage for stability...")
# Set environment variable to disable MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Force PyTorch to use CPU by setting default device
torch.set_default_device("cpu")
device = "cpu"

print(f"Using device: {device}")

# 3. Memory management settings
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 4. LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Load tokenizer first
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 6. Load model with explicit device placement
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=None,  # Don't use device_map, we'll handle device placement manually
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# Explicitly move model to chosen device
print(f"Moving model to {device}...")
model = model.to(device)

# Apply LoRA after moving to device
print("Applying LoRA adapters...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Ensure LoRA model is also on the correct device
model = model.to(device)

# 7. Preprocessing function with memory optimization
def preprocess(batch):
    # Combine prompt and completion
    texts = []
    for p, c in zip(batch["prompt"], batch["reply"]):
        # Add proper formatting
        formatted_text = f"{p}{c}{tokenizer.eos_token}"
        texts.append(formatted_text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=128,  # Keep this small for memory
        padding="max_length",
        return_tensors=None  # Don't return tensors to save memory
    )
    
    # For causal LM, labels are the input_ids
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    
    # Clear intermediate variables
    del texts
    gc.collect()
    
    return tokenized

# 8. Load and preprocess dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": DATA_PATH})

# Check if cached version exists
if os.path.isdir(CACHE_PATH):
    print("Loading cached tokenized dataset...")
    tokenized = load_from_disk(CACHE_PATH)
else:
    print("Tokenizing dataset...")
    tokenized = dataset["train"].map(
        preprocess,
        batched=True,
        batch_size=16,
        remove_columns=["prompt", "reply"],
        num_proc=1,
    )
    print("Saving tokenized dataset to cache...")
    tokenized.save_to_disk(CACHE_PATH)

# Clear original dataset from memory
del dataset
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Dataset size: {len(tokenized)} examples")

# 9. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 10. Training arguments optimized for stability
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    optim="adamw_torch",
    gradient_checkpointing=False,  # Disabled for stability
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    logging_steps=25,
    save_steps=500,
    eval_steps=500,
    eval_strategy="no",
    fp16=False,
    bf16=False,
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_drop_last=True,
    save_total_limit=2,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=None,
    max_grad_norm=1.0,
    warmup_steps=50,
)

# 11. Custom Trainer class with forced CPU operations
class StableTrainer(Trainer):
    def training_step(self, model, inputs):
        # Ensure all inputs are on CPU
        inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Ensure model is on CPU
        if next(model.parameters()).device.type != "cpu":
            model = model.to("cpu")
        
        # Clear cache before each step
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Standard training step
            result = super().training_step(model, inputs)
            return result
        except Exception as e:
            print(f"Training step error: {e}")
            # Clear memory and re-raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
    
    def _save_checkpoint(self, model, trial, metrics=None):
        # Clear cache before saving
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Standard checkpoint saving
        super()._save_checkpoint(model, trial, metrics)

# 12. Trainer setup
print("Setting up trainer...")
trainer = StableTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

# 13. Run training with comprehensive error handling
if __name__ == "__main__":
    try:
        print("Starting training...")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Clear cache before training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer.train()
        
        print("Training completed successfully!")
        
    except torch.cuda.OutOfMemoryError:
        print("CUDA Out of Memory Error - try reducing batch size further")
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            print(f"MPS Error: {e}")
            print("Falling back to CPU-only training")
            # You could add logic here to restart with CPU if needed
        else:
            print(f"Runtime Error: {e}")
            raise e
    except Exception as e:
        print(f"Training error: {e}")
        raise e
    
    finally:
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 14. Save LoRA adapters & tokenizer
print("Saving model and tokenizer...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save LoRA adapters
adapter_path = os.path.join(OUTPUT_DIR, "lora_adapters")
model.save_pretrained(adapter_path)

# Save tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ SFT training complete. Adapters saved to", OUTPUT_DIR)
print(f"✅ LoRA adapters saved to: {adapter_path}")

# Final cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()