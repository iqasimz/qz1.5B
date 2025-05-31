#!/usr/bin/env python3
"""
Nuclear Energy AI: Base vs Fine-tuned Model Comparison
Optimized for Streamlit Cloud deployment
"""

import os
import sys
import warnings

# Suppress torch warnings and set environment variables early
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch,torch.*"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Import streamlit first to initialize properly
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Nuclear Energy AI Comparison",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import torch and other heavy libraries
@st.cache_resource
def import_torch():
    """Import torch with proper error handling for Streamlit Cloud"""
    try:
        import torch
        torch.set_num_threads(2)  # Lower for cloud environment
        return torch
    except Exception as e:
        st.error(f"Failed to import PyTorch: {e}")
        st.stop()

@st.cache_resource
def import_transformers():
    """Import transformers with proper error handling"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        return AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        st.error(f"Failed to import Transformers: {e}")
        st.stop()

# Import libraries
torch = import_torch()
AutoTokenizer, AutoModelForCausalLM = import_transformers()

# Device selection - prioritize CPU for Streamlit Cloud
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = "CUDA GPU"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

# Header
st.title("🔬 Nuclear Energy AI: Base vs Fine-tuned Model Comparison")
st.markdown(f"*Running on Streamlit Cloud using: {device_name}*")

# Model paths
BASE_MODEL_PATH = "gpt2-medium"
FINE_TUNED_MODEL_PATH = "iqasimz/gpt2"  # HuggingFace model ID

# Load models with better error handling for cloud
@st.cache_resource
def load_models():
    """Load models with comprehensive error handling for Streamlit Cloud"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load base model
        status_text.text("Loading base GPT-2 model...")
        progress_bar.progress(25)
        
        base_tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH, 
            use_fast=True,
            cache_dir="/tmp/transformers_cache"  # Use tmp for cloud
        )
        
        progress_bar.progress(50)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            cache_dir="/tmp/transformers_cache",
            torch_dtype=torch.float32,  # Explicit dtype for stability
            low_cpu_mem_usage=True  # Memory optimization
        )
        
        # Load fine-tuned model
        status_text.text("Loading fine-tuned model...")
        progress_bar.progress(75)
        
        fine_tuned_tokenizer = AutoTokenizer.from_pretrained(
            FINE_TUNED_MODEL_PATH,
            use_fast=True,
            cache_dir="/tmp/transformers_cache"
        )
        
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(
            FINE_TUNED_MODEL_PATH,
            cache_dir="/tmp/transformers_cache",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Move models to device
        base_model = base_model.to(device)
        fine_tuned_model = fine_tuned_model.to(device)
        
        # Set to evaluation mode
        base_model.eval()
        fine_tuned_model.eval()
        
        progress_bar.progress(100)
        status_text.text("Models loaded successfully!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return base_tokenizer, base_model, fine_tuned_tokenizer, fine_tuned_model
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("This might be due to memory constraints on Streamlit Cloud")  
        st.info("Try refreshing the page or contact support if the issue persists")
        st.stop()

# Initialize models
if 'models_loaded' not in st.session_state:
    with st.spinner("Initializing AI models... This may take a few minutes on first load."):
        base_tokenizer, base_model, fine_tuned_tokenizer, fine_tuned_model = load_models()
        st.session_state.models_loaded = True
        st.session_state.base_tokenizer = base_tokenizer
        st.session_state.base_model = base_model
        st.session_state.fine_tuned_tokenizer = fine_tuned_tokenizer
        st.session_state.fine_tuned_model = fine_tuned_model
else:
    base_tokenizer = st.session_state.base_tokenizer
    base_model = st.session_state.base_model
    fine_tuned_tokenizer = st.session_state.fine_tuned_tokenizer
    fine_tuned_model = st.session_state.fine_tuned_model

# Generation function optimized for cloud
def generate_response(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_p=0.9):
    """Generate response with cloud-optimized settings"""
    try:
        # Prepare input with length limits for cloud
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=400,  # Reduced for cloud memory
            padding=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 150),  # Cap tokens for cloud
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True  # Enable KV cache for efficiency
            )
        
        response = tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        )
        
        return response.strip()
        
    except torch.cuda.OutOfMemoryError:
        return "GPU memory exceeded. Try reducing max tokens or refresh the page."
    except Exception as e:
        return f"Generation error: {str(e)}"

# Sidebar configuration
st.sidebar.header("⚙️ Generation Settings")
max_tokens = st.sidebar.slider("Max tokens:", 50, 150, 80)  # Reduced range for cloud
temperature = st.sidebar.slider("Temperature:", 0.1, 1.5, 0.8)
top_p = st.sidebar.slider("Top-p:", 0.1, 1.0, 0.9)

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

selected_sample = st.sidebar.selectbox("Choose a sample:", [""] + sample_prompts)

# Main interface
st.header("💬 Ask about Nuclear Energy")

# Input
if selected_sample:
    prompt = st.text_area("Enter your prompt:", value=selected_sample, height=100)
else:
    prompt = st.text_area("Enter your prompt:", placeholder="What do you think about nuclear energy?", height=100)

# Generate button
if st.button("🚀 Generate Responses", type="primary"):
    if prompt.strip():
        # Show processing message
        with st.spinner("🧠 AI models are thinking... This may take 30-60 seconds on Streamlit Cloud."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Base GPT-2")
                with st.spinner("Generating base response..."):
                    base_reply = generate_response(
                        base_model, base_tokenizer, prompt, max_tokens, temperature, top_p
                    )
                st.text_area("Base Model Response:", value=base_reply, height=200, key="base")
                
            with col2:
                st.subheader("⚡ Fine-tuned Model")
                with st.spinner("Generating fine-tuned response..."):
                    tuned_reply = generate_response(
                        fine_tuned_model, fine_tuned_tokenizer, prompt, max_tokens, temperature, top_p
                    )
                st.text_area("Fine-tuned Response:", value=tuned_reply, height=200, key="tuned")
        
        # Analysis
        st.header("📊 Comparison Analysis")
        st.markdown("""
        **Look for these differences:**
        - **Tone & Style**: Is the fine-tuned model more direct or sarcastic?
        - **Nuclear Knowledge**: Does it show deeper technical understanding?
        - **Response Quality**: Which provides more accurate information?
        """)
        
    else:
        st.warning("⚠️ Please enter a prompt first!")

# Footer
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🤖 Base Model**")
    st.markdown("• GPT-2 Medium (355M params)")
    st.markdown("• General knowledge")

with col2:
    st.markdown("**⚡ Fine-tuned Model**")
    st.markdown("• Specialized for nuclear topics")
    st.markdown("• Custom trained dataset")

# Cloud-specific tips
with st.expander("☁️ Streamlit Cloud Tips"):
    st.markdown("""
    **For optimal performance:**
    - First load takes 2-3 minutes (models are downloading)
    - Responses take 30-60 seconds to generate
    - Keep max tokens below 150 to avoid memory issues
    - Refresh page if you encounter memory errors
    - The torch warning messages are harmless and can be ignored
    """)

# Status indicator
if st.session_state.get('models_loaded', False):
    st.sidebar.success("✅ Models Ready")
else:
    st.sidebar.warning("⏳ Loading Models...")