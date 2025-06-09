import streamlit as st
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from peft import PeftModel

# Cache loading of tokenizer and LoRA adapter
def load_model():
    # Load tokenizer (includes special tokens)
    tokenizer = AutoTokenizer.from_pretrained(
        "models/deepseek-finetuned-efficient", trust_remote_code=True
    )
    # Ensure special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})

    # Load base model from original hub
    base_model = Qwen2ForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
        use_cache=True
    )

    # Resize base model embeddings to match our tokenizer
    base_model.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapter on top of the resized base model
    model = PeftModel.from_pretrained(
        base_model,
        "iqasimz/deepseek-finetuned-efficient",
        is_trainable=False,
    )
    model.eval()

    return tokenizer, model

@st.cache_resource
def get_resources():
    return load_model()

tokenizer, model = get_resources()

st.title("DeepSeek LoRA Inference")

user_input = st.text_area("Enter your prompt:")

if st.button("Generate Reply"):
    if not user_input.strip():
        st.warning("Please enter a prompt to generate a reply.")
    else:
        formatted = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(formatted, return_tensors="pt")
        with st.spinner("Generating..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.2,
                top_k=50,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs.input_ids.shape[-1]:]
        reply = tokenizer.decode(generated, skip_special_tokens=True)
        st.write(reply)