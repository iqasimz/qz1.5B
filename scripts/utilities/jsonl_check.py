#!/usr/bin/env python3
import json
from transformers import AutoTokenizer
import numpy as np

# 1) Load your tokenizer (same as your SFT)
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    use_fast=True
)

lengths = []

# 2) Read your JSONL and compute token lengths
with open("data/newlabels.jsonl", "r") as f:
    for line in f:
        ex = json.loads(line)
        text = ex["prompt"] + ex["reply"]
        # encode returns a list of token IDs
        tok_ids = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tok_ids))

lengths = np.array(lengths)

# 3) Print some key statistics
percentiles = [50, 75, 90, 95, 99, 100]
vals = np.percentile(lengths, percentiles)

print("Prompt+reply Token Lengths")
print("-------------------------------")
for p, v in zip(percentiles, vals):
    print(f"{p:>3}th percentile: {int(v)} tokens")
print(f"Maximum length: {int(lengths.max())} tokens")
print(f"Average length: {lengths.mean():.1f} tokens")