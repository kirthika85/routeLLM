import streamlit as st
import os
import openai
from routellm.controller import Controller
import time
from anthropic import Anthropic
import asyncio
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

# Function to get a response from a specific model
async def get_response_from_model(prompt, model_name):
    try:
        if model_name == "gpt-4o":
            # Use OpenAI API for GPT-4o
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content, model_name
        elif model_name == "claude-3-haiku-20240307":
            # Use Anthropic API for Claude
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model="claude-3-haiku@20240307",  # Ensure the correct model version
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content, model_name
        else:
            # Simulate response for other models
            response = f"Simulated response from {model_name}: {prompt} processed."
            return response, model_name
    except Exception as e:
        return f"Error: {e}", None

# Streamlit App
st.title("LLM Router Application")

selected_models = st.multiselect("Select Models", list(models.keys()) + ["RouteLLM Router"])

# Input prompt
prompt = st.text_area("Enter your prompt:", height=100)

# Button to generate response
if st.button("Get Response"):
    columns = st.columns(len(selected_models))
    
    for i, model in enumerate(selected_models):
        if model == "RouteLLM Router":
            response, model_used = get_response(prompt)
            columns[i].write(f"Response from {model_used}:")
            columns[i].write(response)
        else:
            response, model_used = asyncio.run(get_response_from_model(prompt, model))
            columns[i].write(f"Response from {model_used}:")
            columns[i].write(response)
