import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Limit CPU threads for consistency
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Base vs Fine-tuned Model Comparison")

# Paths for base model and LoRA adapters
BASE_MODEL_PATH = "openai-community/gpt2-medium"  # Replace with your base model path
ADAPTERS_PATH = "models/gpt2"

# Load tokenizer and models
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
    fine_tuned_model = PeftModel.from_pretrained(base_model, ADAPTERS_PATH).to(device)
    base_model.eval()
    fine_tuned_model.eval()
    return tokenizer, base_model, fine_tuned_model

tokenizer, base_model, model = load_models()

# Generate response function
def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

# User input
prompt = st.text_input("Enter your prompt:", placeholder="What is nuclear energy?")

if prompt:
    with st.spinner("Generating responses..."):
        base_reply = generate_response(base_model, prompt)
        tuned_reply = generate_response(model, prompt)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Base Model")
        st.write(base_reply)
    with col2:
        st.subheader("Fine-tuned Model")
        st.write(tuned_reply)