import streamlit as st
import os
import openai
from routellm.controller import Controller
import time

with st.spinner("🔄 Mool AI agent Authentication In progress..."):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        st.error("❌ API_KEY not found in environment variables.")
        st.stop()
    time.sleep(5)
st.success("✅ Mool AI Authentication Successful")

if openai.api_key is None:
    st.error("OPENAI_API_KEY environment variable is not set. Please set it before running the app.")
    st.stop()

# Define models and parameters
models = {
    "jamba-1.5-mini": {"vendor": "ai21", "cost_per_prompt": 2e-7, "cost_per_completion": 4e-7},
    "claude-3-haiku-20240307": {"vendor": "anthropic", "cost_per_prompt": 2.5e-7, "cost_per_completion": 1.25e-6},
    "gpt-4o": {"vendor": "openai", "cost_per_prompt": 5e-6, "cost_per_completion": 1.5e-5},
    "llama3.1-8b": {"vendor": "meta-llama", "cost_per_prompt": 1e-7, "cost_per_completion": 1e-7},
}

# Initialize RouteLLM Controller
client = Controller(
    routers=["mf"],  # Specify which router to use
    strong_model="gpt-4-1106-preview",  # Strong model for complex tasks
    weak_model="anyscale/mistralai/Mixtral-8x7B-Instruct-v0.1",  # Weak model for simpler tasks
    config={
        "mf": {
            "checkpoint_path": "routellm/mf_gpt4_augmented"  # Path to model checkpoint
        }
    }
)

# Function to get a response from the router
def get_response(prompt):
    try:
        response = client.chat.completions.create(
            model="router-mf-0.11593",  # Specify the router model with cost threshold
            messages=[{"role": "user", "content": prompt}]
        )
        
        # log the model used
        model_used = "RouteLLM Router"  # Default to router for now
        
        return response.choices[0]["message"]["content"], model_used
    except Exception as e:
        return f"Error: {e}", None

# Function to select a model based on cost parameters
def select_model(prompt, max_cost, willingness_to_pay):
    # Simplified logic: Select a cheaper model if max_cost is low
    if max_cost < 0.01:
        return "jamba-1.5-mini"
    else:
        return "claude-3-haiku-20240307"

# Streamlit App
st.title("LLM Router Application")

# Input prompt
prompt = st.text_area("Enter your prompt:", height=100)

# Button to generate response
if st.button("Get Response"):
    response, model_used = get_response(prompt)
    
    st.write("Response:")
    st.write(response)
    
    if model_used:
        st.write(f"Routed to: {model_used}")
    else:
        st.write("Failed to determine the routed model.")

# Optional: Display model details
if st.checkbox("Show Model Details"):
    st.write("Model Details:")
    for model, details in models.items():
        st.write(f"Model: {model}")
        st.write(f"Vendor: {details['vendor']}")
        st.write(f"Cost per Prompt: {details['cost_per_prompt']}")
        st.write(f"Cost per Completion: {details['cost_per_completion']}")
        st.write("----")
