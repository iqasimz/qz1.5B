#!/usr/bin/env python3
# train_gpt.py
# Fine-tune gpt2-medium on a sarcasm-and-brutality dataset about nuclear energy
# Fixed for macOS with M4 - completely disable MPS

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def main():
    # COMPLETELY disable MPS to force CPU usage
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
    
    # Force CPU usage
    device = torch.device("cpu")
    print(f"Forcing CPU usage, MPS disabled")
    
    # Set torch to use CPU for all operations
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)

    # 1) Load dataset
    print("Loading dataset...")
    data_files = {"train": "data/newlabels.jsonl"}
    dataset = load_dataset("json", data_files=data_files)
    print(f"Dataset loaded: {len(dataset['train'])} examples")

    # 2) Load tokenizer & model
    print("Loading tokenizer and model...")
    model_name = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure a pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    # Load model with explicit device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None  # Don't use automatic device mapping
    )
    
    # Resize embeddings if we added tokens
    original_embeddings = model.get_input_embeddings().weight.size(0)
    if len(tokenizer) > original_embeddings:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings from {original_embeddings} to {len(tokenizer)}")
    
    # Explicitly move ALL model components to CPU
    model = model.cpu()
    for param in model.parameters():
        param.data = param.data.cpu()
        if param.grad is not None:
            param.grad = param.grad.cpu()
    
    print(f"Model loaded and moved to CPU. Device: {next(model.parameters()).device}")

    # 3) Tokenization function
    def tokenize_fn(examples):
        # Combine prompt and reply with EOS tokens
        inputs = []
        for p, r in zip(examples["prompt"], examples["reply"]):
            combined = f"{p}{tokenizer.eos_token}{r}{tokenizer.eos_token}"
            inputs.append(combined)
        
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors=None  # Return lists, not tensors
        )
        
        # Labels are the same as input IDs for causal LM
        model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized = dataset["train"].map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    print("Tokenization complete")

    # 4) Custom data collator to ensure CPU tensors
    class CPUDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            batch = super().__call__(features)
            # Ensure all tensors are on CPU
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.cpu()
            return batch

    data_collator = CPUDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )

    # 5) Training arguments - optimized for CPU
    training_args = TrainingArguments(
        output_dir="models/gpt2",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Very small batch size for CPU
        gradient_accumulation_steps=32,  # Large accumulation to maintain effective batch size
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,  # Reduced warmup steps
        logging_steps=5,   # Frequent logging
        save_steps=200,    # More frequent saves
        eval_steps=None,   # Disable evaluation
        save_total_limit=2,  # Keep only 2 checkpoints
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        use_cpu=True,  # Explicitly use CPU
        no_cuda=True,  # Disable CUDA
        push_to_hub=False,
        report_to=[],  # Disable all reporting
        remove_unused_columns=False,
        prediction_loss_only=True,  # Simplify loss computation
        load_best_model_at_end=False,  # Don't load best model
        metric_for_best_model=None,
        greater_is_better=None,
    )

    # 6) Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 7) Final check - ensure model is on CPU
    print(f"Final model device check: {next(model.parameters()).device}")
    
    # Train and save
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
        
        print("Saving model...")
        trainer.save_model("models/gpt2")
        tokenizer.save_pretrained("models/gpt2")
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Try to save what we have
        try:
            model.save_pretrained("models/gpt2_partial")
            tokenizer.save_pretrained("models/gpt2_partial")
            print("Partial model saved to models/gpt2_partial")
        except:
            print("Could not save partial model")
        raise

if __name__ == "__main__":
    main()