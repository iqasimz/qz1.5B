#!/usr/bin/env python3
import os
import sys
from huggingface_hub import HfApi, upload_folder

# --- CONFIGURE THESE ---
LOCAL_MODEL_DIR = "models/deepseek-argumentanalyst-full"   # your merged checkpoint folder
REPO_ID         = "iqasimz/deepseek-1.5B-argumentanalyst"         # your HF username/repo_name
PRIVATE         = False                         # True to keep the repo private
# -----------------------

api = HfApi()

# 1. Create the repo if it doesn’t already exist
print(f"🔎 Ensuring repo `{REPO_ID}` exists…")
api.create_repo(
    repo_id=REPO_ID,
    private=PRIVATE,
    exist_ok=True
)

# 2. Upload the folder
print(f"📤 Uploading contents of `{LOCAL_MODEL_DIR}` to `https://huggingface.co/{REPO_ID}`…")
upload_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=LOCAL_MODEL_DIR,
    path_in_repo="",
    commit_message="Initial SFT-warmup merged model upload"
)

print("✅ Done! Your model is live at:")
print(f"    https://huggingface.co/{REPO_ID}")