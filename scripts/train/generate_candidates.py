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
PAIRED_PATH     = "data/newlabels.jsonl"
CANDIDATES_OUT  = "data/cand_sft.jsonl"

# 1. Load tokenizer + model with LoRA adapters
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
model = PeftModel.from_pretrained(base_model, ADAPTERS_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Read contexts & generate
with open(PAIRED_PATH) as fin, open(CANDIDATES_OUT, "w") as fout:
    for idx, line in enumerate(fin):
        ex = json.loads(line)
        prompt = ex["prompt"]
        preferred = ex["reply"]
        # tokenize and move inputs to device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # generate one sample from fine-tuned model
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # assemble DPO-style record
        cand = {
            "id": f"example_{idx}",
            "prompt": prompt,
            "response_preferred": preferred,
            "response_non_preferred": gen_text
        }
        fout.write(json.dumps(cand, ensure_ascii=False) + "\n")
print("✓ Candidate generations saved to", CANDIDATES_OUT)