#!/usr/bin/env python3
"""
Diagnostic script to check tokenization and masking
"""

import json
from transformers import AutoTokenizer

def diagnose_tokenization():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    
    # Load a sample from your data
    with open("data/newlabels.jsonl", "r") as f:
        sample = json.loads(f.readline())
    
    print("Sample data:")
    print(f"Prompt: {sample['prompt']}")
    print(f"Reply: {sample['reply']}")
    print("\n" + "="*80 + "\n")
    
    # Format as in training
    full_text = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n{sample['reply']}<|im_end|>"
    print("Formatted text:")
    print(full_text)
    print("\n" + "="*80 + "\n")
    
    # Tokenize
    tokenized = tokenizer(full_text, truncation=True, max_length=192, padding=False, return_tensors=None)
    
    # Check special tokens
    print("Special token IDs:")
    print(f"<|im_start|>: {tokenizer.encode('<|im_start|>', add_special_tokens=False)}")
    print(f"<|im_end|>: {tokenizer.encode('<|im_end|>', add_special_tokens=False)}")
    print(f"assistant: {tokenizer.encode('assistant', add_special_tokens=False)}")
    print("\n" + "="*80 + "\n")
    
    # Find assistant start
    assistant_start_text = "<|im_end|>\n<|im_start|>assistant\n"
    assistant_start_tokens = tokenizer(assistant_start_text, add_special_tokens=False)["input_ids"]
    
    print(f"Looking for assistant start tokens: {assistant_start_tokens}")
    print(f"Token sequence length: {len(tokenized['input_ids'])}")
    
    # Create labels as in training
    labels = [-100] * len(tokenized["input_ids"])
    
    # Find where assistant response starts
    found = False
    for i in range(len(tokenized["input_ids"]) - len(assistant_start_tokens)):
        if tokenized["input_ids"][i:i+len(assistant_start_tokens)] == assistant_start_tokens:
            start_idx = i + len(assistant_start_tokens)
            labels[start_idx:] = tokenized["input_ids"][start_idx:]
            found = True
            print(f"\nFound assistant start at position {i}, labeling from position {start_idx}")
            break
    
    if not found:
        print("\nWARNING: Could not find assistant start tokens!")
    
    # Show tokens and labels
    print("\nFirst 50 tokens and their labels:")
    for i in range(min(50, len(tokenized['input_ids']))):
        token = tokenizer.decode([tokenized['input_ids'][i]])
        label = labels[i]
        label_str = str(label) if label != -100 else "MASKED"
        print(f"{i:3d}: {tokenized['input_ids'][i]:6d} -> '{token:15s}' | Label: {label_str}")
    
    # Check if any tokens are being trained
    trained_tokens = sum(1 for l in labels if l != -100)
    print(f"\nTokens being trained on: {trained_tokens} out of {len(labels)}")
    
    # Alternative approach - simple masking
    print("\n" + "="*80 + "\n")
    print("Alternative approach - mask only the prompt part:")
    
    # Find where reply starts in the formatted text
    user_part = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    user_tokens = tokenizer(user_part, add_special_tokens=False)["input_ids"]
    
    alt_labels = tokenized["input_ids"].copy()
    # Mask everything up to where the assistant reply starts
    alt_labels[:len(user_tokens)] = [-100] * len(user_tokens)
    
    print(f"Alternative masking: training on {sum(1 for l in alt_labels if l != -100)} tokens")
    print("Alternative labels preview:")
    for i in range(min(50, len(alt_labels))):
        token = tokenizer.decode([tokenized['input_ids'][i]])
        label = alt_labels[i]
        label_str = str(label) if label != -100 else "MASKED"
        print(f"{i:3d}: {tokenized['input_ids'][i]:6d} -> '{token:15s}' | Label: {label_str}")

if __name__ == "__main__":
    diagnose_tokenization()