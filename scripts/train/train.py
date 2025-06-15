#!/usr/bin/env python3
"""
Memory-efficient training script for DeepSeek on CPU
Optimized for argumentative text analysis with JSON outputs
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
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "[USR]", "[JSON]"]
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
    """Load and validate dataset"""
    logger.info("Loading dataset...")
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    dataset = load_dataset("json", data_files=data_path)['train']
    
    # Validate first few examples
    logger.info("Validating dataset format...")
    for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
        try:
            # Check if completion is valid JSON
            json.loads(example['completion'])
            logger.info(f"Example {i}: ✓ Valid JSON completion")
        except json.JSONDecodeError as e:
            logger.warning(f"Example {i}: ✗ Invalid JSON completion: {e}")
            
        # Check prompt format
        if "[USR]" in example['prompt'] and "[JSON]" in example['prompt']:
            logger.info(f"Example {i}: ✓ Valid prompt format")
        else:
            logger.warning(f"Example {i}: ✗ Missing [USR] or [JSON] markers")
    
    # Create validation split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    logger.info(f"Dataset: {len(dataset['train'])} train, {len(dataset['test'])} validation")
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=1024):
    """Tokenize with proper JSON handling"""
    
    def tokenize_function(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for prompt, completion in zip(prompts, completions):
            # Handle the case where completion might already be a string
            if isinstance(completion, str):
                try:
                    # Try to parse and reformat JSON for consistency
                    parsed_json = json.loads(completion)
                    formatted_completion = json.dumps(parsed_json, indent=2)
                except json.JSONDecodeError:
                    # If it's not valid JSON, use as-is
                    formatted_completion = completion
            else:
                formatted_completion = json.dumps(completion, indent=2)
            
            # Format text properly
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
    
    # Process in small batches to save memory
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=4,  # Small batch size for memory
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
    data_path = "data/combinedi.jsonl"  # Update this path
    output_dir = "models/deepseek-argumentanalyst"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    dataset = create_dataset(data_path)
    
    # Tokenize with larger max length for JSON outputs
    logger.info("Tokenizing dataset...")
    train_dataset = tokenize_dataset(dataset["train"], tokenizer, max_length=1024)
    eval_dataset = tokenize_dataset(dataset["test"], tokenizer, max_length=1024)
    
    # Check average sequence length
    train_lengths = [len(example['input_ids']) for example in train_dataset.select(range(min(10, len(train_dataset))))]
    logger.info(f"Sample sequence lengths: {train_lengths}")
    logger.info(f"Average length: {sum(train_lengths)/len(train_lengths):.1f} tokens")
    
    # Data collator
    data_collator = EfficientDataCollator(tokenizer.pad_token_id)
    
    # Training arguments optimized for your data
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Very small batches for memory efficiency with longer sequences
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 8
        
        # Training settings
        num_train_epochs=3,  # Fewer epochs for small dataset
        learning_rate=2e-4,  # Slightly lower for JSON structure learning
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=25,  # More frequent evaluation for small dataset
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # CPU settings
        use_cpu=True,
        no_cuda=True,
        dataloader_pin_memory=False,
        fp16=False,  # Disable FP16 for better JSON precision
        dataloader_num_workers=0,
        
        # Memory optimization
        gradient_checkpointing=True,
        optim="adamw_torch",
        adam_epsilon=1e-8,
        
        # Disable unused features
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    logger.info("Starting training for argumentative analysis...")
    logger.info(f"Training on {len(train_dataset)} examples with JSON outputs")
    
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
        
        # Test inference
        logger.info("Testing inference...")
        test_prompt = "[USR]\nClimate change is a serious issue that requires immediate action.\n[JSON]"
        inputs = tokenizer(f"<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test response: {response}")
        
    except MemoryError:
        logger.error("Out of memory! Try these solutions:")
        logger.error("1. Reduce max_length to 512")
        logger.error("2. Reduce gradient_accumulation_steps to 4")
        logger.error("3. Close other applications")
        logger.error("4. Enable swap memory")
        
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