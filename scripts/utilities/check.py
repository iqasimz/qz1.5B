#!/usr/bin/env python3
# debug_generation.py
#
# Load your fine-tuned DeepSeek model/tokenizer, then:
#  1) Print out the special token IDs.
#  2) Show exactly how we format and tokenize a test prompt.
#  3) Generate *greedily* and print every generated token (ID + string).
#
# Usage:
#   python debug_generation.py

import sys
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = "models/deepseek-finetuned-efficient"
DEVICE = "cpu"   # force CPU to match your training environment
# (If you have a GPU and want to test on it, you can set DEVICE="cuda")

# Two test prompts:
TEST_PROMPT_NEW = "What is nuclear energy?"
# Pick a prompt exactly from your JSONL training file (copy‐paste including punctuation).
# Example (you said “Can nuclear power provide consistent energy during...” was repeated):
TEST_PROMPT_KNOWN = "Can nuclear power provide consistent energy during drought seasons?"

MAX_LENGTH = 128  # how many new tokens we’ll allow

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def print_token_info(label: str, toks):
    """
    Print a list of token IDs and their corresponding string tokens.
    """
    print(f"\n--- {label} ---")
    print(" Index │  ID  │  Token")
    print("───────┼──────┼────────────────────────────────────")
    for idx, tid in enumerate(toks):
        tok_str = tokenizer.convert_ids_to_tokens(int(tid))
        print(f"{idx:>6} │ {tid:>4} │ {tok_str!r}")
    print("--------------------\n")


def run_test(prompt_text: str, description: str):
    """
    1) Format with special tokens.
    2) Tokenize and print out IDs+strings.
    3) Generate greedily and show exactly which tokens were generated.
    4) Decode and print final reply.
    """
    print(f"\n========== RUNNING TEST: {description} ==========\n")

    # 1) Format the prompt exactly as in fine-tuning:
    #    <|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n
    formatted = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

    # 2) Tokenize (no truncation here—just print out everything)
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,  # we are manually using the special tokens in `formatted`
    ).to(DEVICE)

    input_ids = inputs["input_ids"][0]      # shape: (L,)
    attn_mask = inputs["attention_mask"][0]  # shape: (L,)

    # 3) Print out the prompt token IDs and token strings, one by one
    print_token_info("Prompt Tokens (ID → string)", input_ids.tolist())

    # 4) Perform a GREEDY generation (no sampling at all)
    #    We force: do_sample=False, top_k=0, top_p=1.0, temperature=1.0
    #    so the model has exactly one choice per step (highest probability).
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.unsqueeze(0),    # shape (1, L)
            attention_mask=attn_mask.unsqueeze(0),
            max_new_tokens=MAX_LENGTH,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,  # so we can inspect the scores if needed
        )

    # The full sequence (prompt + generated) is in outputs.sequences[0]
    full_seq = outputs.sequences[0]  # shape: (L + new_tokens,)

    # 5) Split off the newly generated tokens
    prompt_len = input_ids.size(0)
    generated_ids = full_seq[prompt_len : prompt_len + MAX_LENGTH]

    # 6) Print out each newly generated token (ID → string)
    print_token_info("Generated Tokens (ID → string)", generated_ids.tolist())

    # 7) Decode the generated IDs into a final reply string
    reply = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    print(f"\n>>> Final decoded reply:\n{reply}\n")
    print("========== END OF TEST ==========\n\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) Load tokenizer and model
    print("Loading tokenizer and model from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = Qwen2ForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
        use_cache=True,
    ).to(DEVICE)
    model.eval()

    # 2) Print out the special‐token IDs
    print("\n=== SPECIAL TOKEN IDS ===")
    for special in ["<|im_start|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        print(f"Token {special!r} → ID {tid}")
    print(f"EOS token → ID {tokenizer.eos_token_id}")
    print(f"PAD token → ID {tokenizer.pad_token_id}")
    print("=========================\n")

    # 3) Run Test #1 on a “new/general” prompt
    run_test(TEST_PROMPT_NEW, description="NEW Prompt (unseen during training)")

    # 4) Run Test #2 on a “known” prompt from your training data
    run_test(TEST_PROMPT_KNOWN, description="KNOWN Prompt (from the training set)")

    print("Debugging script complete. Inspect the printed token sequences above.")