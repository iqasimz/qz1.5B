#!/usr/bin/env python3
"""
Detailed diagnostic script to debug the fine-tuned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM
import os

def detailed_diagnostics(model_path: str = "models/deepseek-finetuned"):
    """Run detailed diagnostics on the model"""
    
    print(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    try:
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None
        ).cpu()
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None
        ).cpu()
    
    model.eval()
    print("Model loaded successfully!\n")
    
    # Check special tokens
    print("=== Special Tokens ===")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print()
    
    # Test simple generation
    prompt = "What is nuclear energy?"
    print(f"=== Testing prompt: '{prompt}' ===\n")
    
    # Show tokenization
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    print(f"Formatted prompt:\n{formatted_prompt}")
    print("-" * 50)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=128)
    print(f"Input token IDs: {inputs.input_ids.tolist()[0]}")
    print(f"Input length: {len(inputs.input_ids[0])}")
    print(f"Decoded input: {tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)}")
    print()
    
    # Test 1: Greedy generation with detailed output
    print("=== Test 1: Greedy Generation (step by step) ===")
    with torch.no_grad():
        # Generate one token at a time to see what's happening
        generated_ids = inputs.input_ids[0].tolist()
        
        for i in range(20):  # Generate up to 20 tokens
            # Prepare input
            current_input = torch.tensor([generated_ids]).to(model.device)
            
            # Get logits
            outputs = model(current_input)
            logits = outputs.logits[0, -1, :]
            
            # Get next token
            next_token_id = torch.argmax(logits).item()
            generated_ids.append(next_token_id)
            
            # Decode the new token
            token_text = tokenizer.decode([next_token_id], skip_special_tokens=False)
            print(f"Step {i+1}: Generated token ID {next_token_id} = '{token_text}'")
            
            # Check if we hit EOS
            if next_token_id == tokenizer.eos_token_id:
                print("Hit EOS token, stopping.")
                break
    
    print(f"\nFull generated text: {tokenizer.decode(generated_ids, skip_special_tokens=False)}")
    print()
    
    # Test 2: Check logits distribution
    print("=== Test 2: Checking Logits Distribution ===")
    with torch.no_grad():
        outputs = model(inputs.input_ids)
        logits = outputs.logits[0, -1, :]
        
        # Check for NaN or Inf
        if torch.isnan(logits).any():
            print("WARNING: Logits contain NaN values!")
        if torch.isinf(logits).any():
            print("WARNING: Logits contain Inf values!")
        
        # Get top 10 predictions
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 10)
        
        print("Top 10 predicted tokens:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode([idx.item()], skip_special_tokens=False)
            print(f"  {i+1}. Token ID {idx.item()} ('{token}'): {prob.item():.4f}")
    
    print()
    
    # Test 3: Try with different prompts
    print("=== Test 3: Multiple Prompts ===")
    test_prompts = [
        "What is",
        "The sun is",
        "Nuclear energy",
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    ]
    
    for test_prompt in test_prompts:
        print(f"\nPrompt: '{test_prompt}'")
        inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            print(f"Generated: '{generated_text}'")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test 4: Check if model is in training mode
    print("\n=== Test 4: Model State ===")
    print(f"Model training mode: {model.training}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test 5: Try the exact training format
    print("\n=== Test 5: Exact Training Format ===")
    # This matches exactly what was used in training
    user_input = "What is nuclear energy?"
    full_text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n Nuclear energy is the heat we harvest when we violently split heavy atoms—think glow-in-the-dark fireworks under strict supervision, not an uncontrolled bomb.<|im_end|>"
    
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True)
    print(f"Training example length: {len(inputs.input_ids[0])}")
    
    # Try to continue from just the prompt part
    prompt_only = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt_only, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Generated with exact format: {generated}")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/deepseek-finetuned"
    detailed_diagnostics(model_path)