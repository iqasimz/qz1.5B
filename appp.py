import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configure Streamlit page
st.set_page_config(page_title="QZDS Assistant", layout="centered")

# Load model and tokenizer (only once)
@st.cache_resource
def load_model():
    base_model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    lora_adapter_path = "models/qzds1.5b-lora"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.to("cpu")
    return tokenizer, model

tokenizer, model = load_model()

# Define response generation
def generate_response(prompt, max_length=64, temperature=0.7, top_p=0.95):
    input_text = f"Human: {prompt}\n\nAssistant:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        except RuntimeError:
            # Fallback to greedy decoding if sampling fails
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.split("Assistant:")[-1].strip()

# Streamlit UI
st.title("🧠 QZDS Assistant (Fine-Tuned)")
user_input = st.text_area("Enter your prompt:", height=120)

if st.button("Generate Reply"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            response = generate_response(user_input.strip())
            st.markdown("**Assistant:**")
            st.success(response)
    else:
        st.warning("Please enter a prompt to get a response.")