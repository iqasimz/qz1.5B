#!/usr/bin/env python3
# train_lora.py
# LoRA fine-tuning script with PEFT (Parameter Efficient Fine-Tuning)

import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Dict, List

def load_and_validate_dataset(file_path: str) -> List[Dict]:
    """Load and validate the training dataset"""
    print(f"Loading dataset from {file_path}...")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                # Validate required fields
                if "prompt" not in item or "reply" not in item:
                    print(f"Warning: Line {line_num} missing prompt or reply")
                    continue
                
                # Validate content
                if not item["prompt"].strip() or not item["reply"].strip():
                    print(f"Warning: Line {line_num} has empty prompt or reply")
                    continue
                
                data.append({
                    "prompt": item["prompt"].strip(),
                    "reply": item["reply"].strip()
                })
                
            except json.JSONDecodeError:
                print(f"Warning: Line {line_num} is not valid JSON")
                continue
    
    print(f"Loaded {len(data)} valid examples")
    return data

def create_training_text(prompt: str, reply: str, tokenizer) -> str:
    """Create properly formatted training text"""
    return f"Human: {prompt}\n\nAssistant: {reply}{tokenizer.eos_token}"

def print_trainable_parameters(model):
    """Print the number of trainable parameters"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}")

def main():
    # Force CPU usage to avoid MPS issues on Mac
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("cpu")
    print(f"Using device: {device} (forced CPU)")
    
    # Set environment variables for CPU optimization
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    torch.set_num_threads(4)
    
    # Disable MPS backend completely
    torch.backends.mps.is_available = lambda: False
    
    # Load dataset
    dataset_path = "data/newlabels.jsonl"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file {dataset_path} not found!")
        return
    
    raw_data = load_and_validate_dataset(dataset_path)
    if not raw_data:
        print("No valid training data found!")
        return
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    # Load model with CPU-specific settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU stability
        device_map=None,  # Don't use auto device mapping
        low_cpu_mem_usage=True
    )
    
    # Explicitly move to CPU
    model = model.to(device)
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    # Don't use prepare_model_for_kbit_training on CPU as it's for quantization
    print("Skipping quantization preparation for CPU training")
    
    # Configure LoRA for CPU training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Reduced rank for CPU efficiency
        lora_alpha=16,  # Reduced alpha proportionally
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],  # Target modules for LoRA (attention and MLP layers)
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Don't adapt bias parameters
        use_rslora=False,  # Use RSLoRA (Rank-Stabilized LoRA)
        modules_to_save=None,  # Additional modules to save (not LoRA adapted)
    )
    
    # Apply LoRA to the model
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    # Prepare training data
    print("Preparing training data...")
    formatted_texts = []
    for item in raw_data:
        formatted_text = create_training_text(item["prompt"], item["reply"], tokenizer)
        formatted_texts.append(formatted_text)
    
    dataset = Dataset.from_dict({"text": formatted_texts})
    
    # Tokenization function
    def tokenize_function(examples):
        """Tokenize with proper padding and truncation"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=256,  # Shorter sequences for CPU efficiency
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = []
        for input_ids in tokenized["input_ids"]:
            labels = input_ids.copy()
            # Replace pad token ids with -100 so they're ignored in loss
            labels = [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in labels]
            tokenized["labels"].append(labels)
        
        return tokenized
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None
    )
    
    # Training arguments - optimized for LoRA
    output_dir = "models/qzds1.5b-lora"
    os.makedirs(output_dir, exist_ok=True)
    
    total_examples = len(tokenized_dataset)
    batch_size = 1  # Small batch size for CPU
    gradient_accumulation = 8  # Reduced for CPU efficiency
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=2,  # Fewer epochs for CPU training
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=5e-5,  # More conservative learning rate for CPU
        weight_decay=0.01,
        warmup_steps=min(50, total_examples // 20),
        
        # Memory optimization for CPU
        fp16=False,  # No fp16 on CPU
        bf16=False,
        dataloader_pin_memory=False,  # Don't pin memory on CPU
        dataloader_num_workers=0,
        
        # Logging and saving
        logging_steps=max(1, total_examples // 20),
        save_steps=max(10, total_examples // 10),
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="no",
        
        # Other settings
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=[],
        
        # Force CPU usage
        use_cpu=True,
        
        # Optimizer
        optim="adamw_torch",
        
        # Save settings
        save_safetensors=True,
    )
    
    print(f"Training configuration:")
    print(f"  - Total examples: {total_examples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gradient accumulation: {gradient_accumulation}")
    print(f"  - Effective batch size: {batch_size * gradient_accumulation}")
    print(f"  - LoRA rank: {lora_config.r}")
    print(f"  - LoRA alpha: {lora_config.lora_alpha}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting LoRA training on CPU...")
    try:
        # Clear any cached memory
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        trainer.train()
        print("Training completed successfully!")
        
        # Save the LoRA adapter
        print("Saving LoRA adapter and tokenizer...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"LoRA adapter saved to {output_dir}")
        
        # Save training configuration
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "base_model": model_name,
                "lora_config": {
                    "r": lora_config.r,
                    "lora_alpha": lora_config.lora_alpha,
                    "target_modules": lora_config.target_modules,
                    "lora_dropout": lora_config.lora_dropout,
                },
                "training_examples": total_examples,
                "epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
            }, indent=2)
        print(f"Training configuration saved to {config_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        # Save partial adapter
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Partial LoRA adapter saved to {output_dir}")
        except Exception as save_error:
            print(f"Could not save adapter: {save_error}")
        raise

if __name__ == "__main__":
    main()