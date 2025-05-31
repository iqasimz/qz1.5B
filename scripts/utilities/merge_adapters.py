from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

BASE = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"        # or your local base folder
ADAPTERS = "checkpoints/dpo/lora_adapters"
OUT_DIR = "models/merged_model"

# 1. Load base and adapter-wrapped model
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base_model, ADAPTERS)

# 2. Merge & unload adapters into the core weights
merged = model.merge_and_unload()

# 3. Save merged model and tokenizer
os.makedirs(OUT_DIR, exist_ok=True)
merged.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"Merged model saved to {OUT_DIR}")