import os
import base64
from io import BytesIO

import streamlit as st
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def model_infer(text, image, model_name):
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_KEY"),
    )

    base64_image = encode_image_to_base64(image)

    completion = client.chat.completions.create(
        model=f"Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic:hyperbolic",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
    )

    return completion.choices[0].message.content

# List of available models
MODEL_LIST = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
    "openai/gpt-oss-120b-turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "FLUX.1-dev",
    "StableDiffusion",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1",
    "moonshotai/Kimi-K2-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-235B-A22B",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "NousResearch/Hermes-3-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "mistralai/Pixtral-12B-2409",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "TTS",
    "openai/gpt-oss-20b",
    "deepseek-ai/DeepSeek-R1-0528",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Thinking"
]

# Sidebar for Huggingface API key input
with st.sidebar:
    st.header("Configuration")
    hf_key = st.text_input(
        "Huggingface API Key",
        type="password",
        value=os.getenv("HF_KEY") or st.session_state.get("hf_key", ""),
        help="Paste your Huggingface Inference API key here.",
    )
    if hf_key:
        st.session_state["hf_key"] = hf_key
        os.environ["HF_KEY"] = hf_key
    st.markdown(
        """
        <small>
        Your API key is only stored in your session and never sent anywhere except to Huggingface.
        </small>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Divergent Model Evaluator", layout="centered")

st.markdown(
    """
    <style>
    .chat-container {
        
    }
    .chat-bubble {
        background: #fff;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Center the title using markdown and HTML
st.markdown(
    "<h1 style='text-align: center;'>Divergent-GPFree</h1>",
    unsafe_allow_html=True,
)


# Display chat history
for entry in st.session_state.history:
    st.markdown(f'<div class="chat-bubble"><b>You:</b> {entry["user"]}', unsafe_allow_html=True)
    st.image(entry["image"], width=200)
    st.markdown(f'<div class="chat-bubble" style="background:#e6f4ea;"><b>{entry["model"]}:</b> {entry["response"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("Enter your message", key="user_input")
    with col2:
        selected_model = st.selectbox("Model", MODEL_LIST, key="model_select")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image_upload")
    submitted = st.form_submit_button("Send")

if submitted and user_input and uploaded_image:
    image = Image.open(uploaded_image)
    response = model_infer(user_input, image, selected_model)
    st.session_state.history.append(
        {"user": user_input, "image": image.copy(), "model": selected_model, "response": response}
    )
    # After first response, rerun to show input at bottom
    # st.experimental_rerun()
