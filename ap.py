import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Limit CPU threads for consistency (especially important for CPU deployment)
torch.set_num_threads(4)

# Device selection - prioritize CPU since we had MPS issues during training
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "CUDA GPU"
elif torch.backends.mps.is_available():
    # You can try MPS for inference (it's more stable than training)
    device = torch.device("mps")
    device_name = "Apple Silicon (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

st.title("🔬 Nuclear Energy AI: Base vs Fine-tuned Model Comparison")
st.markdown(f"*Running on: {device_name}*")

# Paths for models
BASE_MODEL_PATH = "gpt2-medium"  # Base model from Hugging Face
FINE_TUNED_MODEL_PATH = "models/gpt2"  # Your fine-tuned model

# Check if fine-tuned model exists
if not os.path.exists(FINE_TUNED_MODEL_PATH):
    st.error(f"Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}")
    st.info("Please make sure your trained model is saved in the 'models/gpt2' directory")
    st.stop()

# Load tokenizer and models
@st.cache_resource
def load_models():
    try:
        # Load base model
        st.info("Loading base model...")
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
        
        # Load fine-tuned model
        st.info("Loading fine-tuned model...")
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH, use_fast=True)
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_MODEL_PATH)
        
        # Move models to device
        base_model = base_model.to(device)
        fine_tuned_model = fine_tuned_model.to(device)
        
        # Set to evaluation mode
        base_model.eval()
        fine_tuned_model.eval()
        
        st.success("Models loaded successfully!")
        return base_tokenizer, base_model, fine_tuned_tokenizer, fine_tuned_model
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Load models
base_tokenizer, base_model, fine_tuned_tokenizer, fine_tuned_model = load_models()

# Generate response function
def generate_response(model, tokenizer, prompt, max_tokens=100):
    try:
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get the length of input tokens for later slicing
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        # Decode only the new tokens (excluding the input prompt)
        response = tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        )
        
        return response.strip()
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Sidebar for configuration
st.sidebar.header("⚙️ Generation Settings")
max_tokens = st.sidebar.slider("Max tokens to generate:", 50, 200, 100)
temperature = st.sidebar.slider("Temperature (creativity):", 0.1, 2.0, 0.8)
top_p = st.sidebar.slider("Top-p (nucleus sampling):", 0.1, 1.0, 0.9)

# Sample prompts
st.sidebar.header("🎯 Sample Prompts")
sample_prompts = [
    "What is nuclear energy?",
    "Explain nuclear power plants",
    "Are nuclear reactors safe?",
    "Nuclear waste disposal methods",
    "Future of nuclear energy",
    "Nuclear fusion vs fission"
]

selected_sample = st.sidebar.selectbox("Choose a sample prompt:", [""] + sample_prompts)

# Main interface
st.header("💬 Ask about Nuclear Energy")
st.markdown("Enter a prompt to see how the base model compares to your fine-tuned model:")

# Input field
if selected_sample:
    prompt = st.text_area("Enter your prompt:", value=selected_sample, height=100)
else:
    prompt = st.text_area("Enter your prompt:", placeholder="What do you think about nuclear energy?", height=100)

# Generate button
if st.button("🚀 Generate Responses", type="primary"):
    if prompt.strip():
        with st.spinner("Generating responses... This may take a moment on CPU."):
            # Generate responses
            base_reply = generate_response(base_model, base_tokenizer, prompt, max_tokens)
            tuned_reply = generate_response(fine_tuned_model, fine_tuned_tokenizer, prompt, max_tokens)
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Base GPT-2 Medium")
            st.markdown("*Standard pre-trained model*")
            st.text_area("Response:", value=base_reply, height=200, key="base_response")
            
        with col2:
            st.subheader("⚡ Fine-tuned Model")
            st.markdown("*Trained on nuclear energy dataset*")
            st.text_area("Response:", value=tuned_reply, height=200, key="tuned_response")
        
        # Analysis section
        st.header("📊 Comparison Analysis")
        st.markdown("**Key Differences to Look For:**")
        st.markdown("""
        - **Tone**: Does the fine-tuned model show more sarcasm or brutality as intended?
        - **Technical Detail**: Does it provide more specific nuclear energy information?
        - **Style**: How does the writing style differ between models?
        - **Accuracy**: Which model provides more accurate information?
        """)
        
    else:
        st.warning("Please enter a prompt first!")

# Footer information
st.markdown("---")
st.markdown("### 🔧 Model Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Base Model:**")
    st.markdown("- GPT-2 Medium (355M parameters)")
    st.markdown("- Pre-trained on diverse text")
    st.markdown("- General language understanding")

with col2:
    st.markdown("**Fine-tuned Model:**")
    st.markdown("- Based on GPT-2 Medium")
    st.markdown("- Fine-tuned on nuclear energy dataset")
    st.markdown("- Specialized for nuclear topics")

# Performance tips
with st.expander("💡 Performance Tips"):
    st.markdown("""
    **For better performance:**
    - Use shorter prompts for faster generation
    - Lower max_tokens for quicker responses
    - If using CPU, expect slower generation times
    - Try different temperature settings for varied creativity
    
    **Troubleshooting:**
    - If you get memory errors, try restarting the app
    - For MPS issues on Mac, the app will fall back to CPU
    - Make sure your fine-tuned model is in the correct directory
    """)