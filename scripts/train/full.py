#!/usr/bin/env python3
"""
Full fine-tuning script for DeepSeek - Better for JSON generation
Higher memory usage but much better results for structured outputs
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Force garbage collection
gc.collect()

class MemoryCallback(TrainerCallback):
    """Monitor memory usage"""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return control

def setup_environment():
    """Configure for memory usage"""
    # Force CPU for all operations
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    
    # Limit threads
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    
    # Seed for reproducibility
    torch.manual_seed(42)
    
    logger.info("Environment configured for full fine-tuning")

def load_model_and_tokenizer(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """Load model for full fine-tuning"""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add special tokens
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "[USR]", "[JSON]"]
    })
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model for full fine-tuning...")
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
        use_cache=False,
    )
    
    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize embeddings for new tokens
    special_ids = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
    with torch.no_grad():
        for tok_id in special_ids:
            nn.init.normal_(model.model.embed_tokens.weight[tok_id], mean=0.0, std=0.02)

    # Move to CPU and enable gradient checkpointing
    model = model.cpu()
    model.gradient_checkpointing_enable()
    
    # Make all parameters trainable (full fine-tuning)
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")
    
    gc.collect()
    return model, tokenizer

def create_dataset(data_path):
    """Load and validate dataset"""
    logger.info("Loading dataset...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    dataset = load_dataset("json", data_files=data_path)['train']
    
    # Validate examples
    logger.info("Validating dataset...")
    valid_count = 0
    for example in dataset:
        try:
            json.loads(example['completion'])
            valid_count += 1
        except:
            continue
    
    logger.info(f"Found {valid_count}/{len(dataset)} valid examples")
    
    # Create validation split
    dataset = dataset.train_test_split(test_size=0.15, seed=42)
    
    logger.info(f"Dataset: {len(dataset['train'])} train, {len(dataset['test'])} validation")
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=1024):
    """Tokenize dataset"""
    def tokenize_function(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for prompt, completion in zip(prompts, completions):
            # Clean and format completion
            if isinstance(completion, str):
                try:
                    parsed_json = json.loads(completion)
                    formatted_completion = json.dumps(parsed_json, separators=(',', ':'))  # Compact JSON
                except:
                    formatted_completion = completion
            else:
                formatted_completion = json.dumps(completion, separators=(',', ':'))
            
            # Format text
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{formatted_completion}<|im_end|>"
            user_part = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize
            full_tokens = tokenizer(text, truncation=True, max_length=max_length, padding=False)
            user_tokens = tokenizer(user_part, truncation=True, max_length=max_length, padding=False)
            
            # Create labels (mask user input)
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
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=8,
        remove_columns=dataset.column_names
    )
    
    return tokenized

class DataCollator:
    """Data collator for full fine-tuning"""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features):
        max_length = max(len(f["input_ids"]) for f in features)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for f in features:
            padding_length = max_length - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * padding_length)
            attention_mask.append(f["attention_mask"] + [0] * padding_length)
            labels.append(f["labels"] + [-100] * padding_length)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def main():
    setup_environment()
    
    # Configuration
    data_path = "data/combinedi.jsonl"
    output_dir = "models/deepseek-argumentanalyst-full"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    dataset = create_dataset(data_path)
    
    # Tokenize
    logger.info("Tokenizing dataset...")
    train_dataset = tokenize_dataset(dataset["train"], tokenizer, max_length=512)
    eval_dataset = tokenize_dataset(dataset["test"], tokenizer, max_length=512)
    
    # Data collator
    data_collator = DataCollator(tokenizer.pad_token_id)
    
    # Training arguments for full fine-tuning
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Smaller batches for memory
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch size = 4
        
        # Training settings optimized for JSON
        num_train_epochs=8,             # More epochs for full training
        learning_rate=1e-5,             # Lower LR for full fine-tuning
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="steps", 
        save_steps=20,
        save_total_limit=3,
        
        # Logging
        logging_steps=5,
        logging_first_step=True,
        
        # CPU settings
        use_cpu=True,
        no_cuda=True,
        dataloader_pin_memory=False,
        fp16=False,  # Disable for JSON precision
        dataloader_num_workers=0,
        
        # Memory optimization
        gradient_checkpointing=True,
        optim="adamw_torch",
        adam_epsilon=1e-8,
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Disable unused features
        push_to_hub=False,
        report_to="none",
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
    logger.info("Starting full fine-tuning...")
    logger.info("This will take longer but should give much better JSON results")
    
    try:
        gc.collect()
        
        # Log memory usage
        import psutil
        process = psutil.Process(os.getpid())
        logger.info(f"Memory usage before training: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        
        # Train
        trainer.train()
        
        # Save full model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Full fine-tuning complete! Model saved to {output_dir}")
        
        # Test the model
        logger.info("Testing trained model...")
        test_prompt = "[USR]\nClimate change requires action.\n[JSON]"
        inputs = tokenizer(f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test response: {response}")
        
    except MemoryError:
        logger.error("Out of memory! Try:")
        logger.error("1. Reduce max_length to 256")
        logger.error("2. Reduce gradient_accumulation_steps to 2") 
        logger.error("3. Use gradient_checkpointing_every_n=2")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        gc.collect()

if __name__ == "__main__":
    main()