#!/usr/bin/env python3
"""
Memory-efficient training script for DeepSeek on CPU
Optimized for low memory usage
"""

import os
import gc
import json
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Qwen2ForCausalLM,
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset
import logging
from peft import LoraConfig, get_peft_model, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Force garbage collection
gc.collect()

class MemoryCallback(TrainerCallback):
    """Monitor memory usage"""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return control

def setup_environment():
    """Configure for minimal memory usage"""
    # Disable MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

    # Force CPU for all operations
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    
    # Limit threads to reduce memory overhead
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    
    # Set memory-efficient settings
    torch.set_grad_enabled(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Seed for reproducibility
    torch.manual_seed(42)
    
    logger.info("Environment configured for low memory usage")

def load_model_and_tokenizer(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Load model with minimal memory footprint"""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure special tokens are recognized as single tokens
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
    })
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model with low memory settings...")
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,  # Enable low memory mode
        device_map=None,
        use_cache=False,  # Disable cache for training
    )
    
    # Resize model embeddings to account for newly added special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Initialize embeddings for new special tokens only
    special_ids = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
    with torch.no_grad():
        for tok_id in special_ids:
            nn.init.normal_(model.model.embed_tokens.weight[tok_id], mean=0.0, std=0.02)

    # Move to CPU and enable gradient checkpointing
    model = model.cpu()
    model.gradient_checkpointing_enable()  # Trade compute for memory
    
    # Apply LoRA for efficient fine-tuning
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, peft_config)
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Clear any unnecessary references
    gc.collect()
    
    return model, tokenizer

def create_dataset(data_path):
    """Load full dataset"""
    logger.info("Loading full dataset...")
    
    dataset = load_dataset("json", data_files=data_path)['train']
    
    # Create validation split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    logger.info(f"Dataset: {len(dataset['train'])} train, {len(dataset['test'])} validation")
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=128):
    """Tokenize with minimal memory usage"""
    
    def tokenize_function(examples):
        prompts = examples["prompt"]
        replies = examples["reply"]
        
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for prompt, reply in zip(prompts, replies):
            # Format text
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>"
            user_part = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize
            full_tokens = tokenizer(text, truncation=True, max_length=max_length)
            user_tokens = tokenizer(user_part, truncation=True, max_length=max_length)
            
            # Create labels
            labels = full_tokens["input_ids"].copy()
            labels[:len(user_tokens["input_ids"])] = [-100] * len(user_tokens["input_ids"])
            
            all_input_ids.append(full_tokens["input_ids"])
            all_attention_masks.append(full_tokens["attention_mask"])
            all_labels.append(labels)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels
        }
    
    # Process in small batches to save memory
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=8,  # Very small batch size for memory
        remove_columns=dataset.column_names
    )
    
    return tokenized

class EfficientDataCollator:
    """Memory-efficient data collator"""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features):
        # Get max length in this batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        # Pad all sequences
        input_ids = []
        attention_mask = []
        labels = []
        
        for f in features:
            # Pad input_ids
            padding_length = max_length - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * padding_length)
            
            # Pad attention_mask
            attention_mask.append(f["attention_mask"] + [0] * padding_length)
            
            # Pad labels
            labels.append(f["labels"] + [-100] * padding_length)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def main():
    # Setup
    setup_environment()
    
    # Configuration
    data_path = "data/newlabels.jsonl"
    output_dir = "models/deepseek-finetuned-efficient"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load full dataset
    dataset = create_dataset(data_path)
    
    # Tokenize with 128 max length
    logger.info("Tokenizing dataset...")
    train_dataset = tokenize_dataset(dataset["train"], tokenizer, max_length=128)
    eval_dataset = tokenize_dataset(dataset["test"], tokenizer, max_length=128)
    
    # Data collator
    data_collator = EfficientDataCollator(tokenizer.pad_token_id)
    
    # Training arguments for low memory with full dataset
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Very small batches for memory efficiency
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch size = 16
        
        # Training settings
        num_train_epochs=5,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        
        # Save memory by evaluating less frequently
        eval_strategy="steps",
        eval_steps=100,  # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        
        # Logging
        logging_steps=20,
        logging_first_step=True,
        
        # CPU settings
        use_cpu=True,
        no_cuda=True,
        dataloader_pin_memory=False,
        fp16=True,
        dataloader_num_workers=0,
        
        # Memory optimization
        gradient_checkpointing=True,
        optim="adamw_torch",
        adam_epsilon=1e-8,
        
        # Disable unused features
        push_to_hub=False,
        hub_strategy="end",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        report_to="none",
        
        # Additional memory saving
        remove_unused_columns=True,
        label_names=["labels"],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[MemoryCallback()],
    )
    
    # Train
    logger.info("Starting training with full dataset (448 samples, 128 tokens)...")
    logger.info("This will take a while on CPU. Monitor memory usage.")
    
    try:
        # Clear memory before training
        gc.collect()
        
        # Log memory usage
        import psutil
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage before training: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        # Train
        trainer.train()
        
        # Save LoRA adapter and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training complete! Model saved to {output_dir}")
        
    except MemoryError:
        logger.error("Out of memory! Try these solutions:")
        logger.error("1. Close other applications")
        logger.error("2. Reduce gradient_accumulation_steps to 8")
        logger.error("3. Enable swap memory on your system")
        logger.error("4. Use a cloud instance with more RAM")
        
        # Emergency save
        try:
            model.save_pretrained(f"{output_dir}_emergency")
            tokenizer.save_pretrained(f"{output_dir}_emergency")
            logger.info("Emergency save completed")
        except:
            pass
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Try to save what we have
        try:
            model.save_pretrained(f"{output_dir}_emergency")
            tokenizer.save_pretrained(f"{output_dir}_emergency")
            logger.info("Emergency save completed")
        except:
            pass
        raise
    
    finally:
        # Clean up
        gc.collect()

if __name__ == "__main__":
    main()