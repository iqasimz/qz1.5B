import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel

# Paths
BASE_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"         # your base model dir
ADAPTERS_PATH   = "checkpoints/sft_warmup/lora_adapters"
PAIRED_PATH     = "data/labels.jsonl"
CANDIDATES_OUT  = "data/cand_sft.jsonl"

# 1. Load tokenizer + model with LoRA adapters
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
model = PeftModel.from_pretrained(base_model, ADAPTERS_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Read contexts & generate
with open(PAIRED_PATH) as fin, open(CANDIDATES_OUT, "w") as fout:
    for line in fin:
        ex = json.loads(line)
        # build prompt string from dialogue up to last user turn
        convo = ""
        for turn in ex["dialogue"]:
            speaker = "User" if turn["speaker"] == "user" else "Assistant"
            convo += f"{speaker}: {turn['text']}\n"
        convo += "Assistant:"
        # tokenize and move inputs to device
        inputs = tokenizer(convo, return_tensors="pt", truncation=True, max_length=512).to(device)
        # generate one sample
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # extract only the model’s reply (after “Assistant:”)
        reply = gen_text.split("Assistant:")[-1].strip()
        cand = {
            "id": ex["id"],
            "dialogue": ex["dialogue"],
            "response_model": reply,
            "response_preferred": ex["response_preferred"],
            "response_non_preferred": ex["response_non_preferred"]
        }
        fout.write(json.dumps(cand, ensure_ascii=False) + "\n")
print("✓ Candidate generations saved to", CANDIDATES_OUT)