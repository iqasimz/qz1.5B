import os
# Workaround MPS memory limit on Mac: disable high water mark
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig
from trl import DPOTrainer, DPOConfig

# 1. Paths & settings
# Use your locally SFT-fine-tuned model as the policy base
BASE_MODEL_PATH   = "checkpoints/sft_warmup"
# Reference model remains the original upstream
REF_BASE_MODEL    = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTERS_PATH     = "checkpoints/sft_warmup/lora_adapters"
PREF_PAIRS_PATH   = "data/preference_pairs.jsonl"
OUTPUT_DIR        = "checkpoints/dpo"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# 2. Load tokenizer + base model + adapters
# Load tokenizer from the original upstream model (in case local SFT checkpoint lacks tokenizer files)
tokenizer = AutoTokenizer.from_pretrained(REF_BASE_MODEL, use_fast=True)
# Ensure tokenizer has a pad token for causal models
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Determine appropriate torch dtype based on device
if torch.cuda.is_available():
    load_dtype = torch.float16
else:
    load_dtype = torch.float32  # MPS and CPU use float32

# Policy base: load original upstream then apply local LoRA adapters

# Apply LoRA configuration and load adapters
base_model = AutoModelForCausalLM.from_pretrained(
    REF_BASE_MODEL,
    torch_dtype=load_dtype,
)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
# Load your LoRA weights from the local directory under the adapter name 'lora'
model.load_adapter(ADAPTERS_PATH, adapter_name="lora")
# Activate the 'lora' adapter for training
model.set_adapter("lora")
# Debug: ensure LoRA adapter parameters are trainable
model.print_trainable_parameters()
trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
if trainable_count == 0:
    raise RuntimeError(
        f"No trainable parameters detected. Please verify ADAPTERS_PATH: {ADAPTERS_PATH}"
    )
model.to(DEVICE)
# Sync pad token
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Load and preprocess preference-pair dataset
ds = load_dataset("json", data_files={"train": PREF_PAIRS_PATH})

def preprocess_raw(ex):
    prompt = ""
    for turn in ex.get("dialogue", []):
        if isinstance(turn, dict):
            role = "User" if turn.get("speaker", "").lower() == "user" else "Assistant"
            text = turn.get("text", "")
        else:
            if ": " in turn:
                role, text = turn.split(": ", 1)
            else:
                role, text = "Assistant", turn
        prompt += f"{role}: {text}\n"
    prompt += "Assistant:"
    return {
        "prompt": prompt,
        "chosen": ex.get("response_preferred", ""),
        "rejected": ex.get("response_non_preferred", ""),
    }

ds = ds.map(preprocess_raw, remove_columns=ds["train"].column_names)
train_dataset = ds["train"]

# 4. DPO configuration (with SFT-style training arguments)
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.5,
    optim="adamw_torch",
    learning_rate=2e-4,
    weight_decay=0.01,
    max_length=128,
    max_prompt_length=32,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=False,
    num_train_epochs=1,
    max_steps=100,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    evaluation_strategy="no",
    bf16=(DEVICE == "cuda"),
)

# 5. Load reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    REF_BASE_MODEL,
    torch_dtype=load_dtype,
).to(DEVICE)
ref_model.eval()

# 6. Initialize and run DPO trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# 7. Save adapters & tokenizer
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapters"))
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ DPO training complete. Adapters saved to", OUTPUT_DIR)