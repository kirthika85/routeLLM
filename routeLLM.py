import streamlit as st
import os
import openai
from routellm.controller import Controller
import time
from anthropic import Anthropic

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
    strong_model="gpt-4o",  # Strong model for complex tasks
    weak_model="claude-3-haiku-20240307",  # Weak model for simpler tasks
    config={
        "mf": {
            "checkpoint_path": "routellm/mf_gpt4_augmented"  # Path to model checkpoint
        }
    }
)

# Function to calculate cost
def calculate_cost(model_name, input_tokens, output_tokens):
    if model_name == "gpt-4o":
        return (input_tokens + output_tokens) * 5 / 1e6
    elif model_name == "claude-3-haiku-20240307":
        return (input_tokens * 0.8 + output_tokens * 4) / 1e6
    elif model_name == "RouteLLM Router":
        strong_model_cost = (input_tokens + output_tokens) * 5 / 1e6
        weak_model_cost = (input_tokens * 0.8 + output_tokens * 4) / 1e6
        return (strong_model_cost + weak_model_cost) / 2
    else:
        return 0

# Function to get a response from the RouteLLM router
def get_response(prompt):
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="router-mf-0.11593",  # Specify the router model with cost threshold
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        latency = end_time - start_time
        cost = calculate_cost("RouteLLM Router", len(prompt), len(response.choices[0]["message"]["content"]))
        return response.choices[0]["message"]["content"], "RouteLLM Router", latency, cost
    except Exception as e:
        return f"Error: {e}", None, None, None

# Function to get a response from a specific model (e.g., GPT-4o, Claude)
def get_response_from_model(prompt, model_name):
    try:
        start_time = time.time()
        if model_name == "gpt-4o":
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )
            end_time = time.time()
            latency = end_time - start_time
            cost = calculate_cost(model_name, len(prompt), len(response.choices[0].message.content))
            return response.choices[0].message.content, model_name, latency, cost
        elif model_name == "claude-3-haiku-20240307":
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model="claude-3-haiku@20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            end_time = time.time()
            latency = end_time - start_time
            cost = calculate_cost(model_name, len(prompt), len(message.content))
            return message.content, model_name, latency, cost
        else:
            end_time = time.time()
            latency = end_time - start_time
            cost = 0
            return f"Simulated response from {model_name}: {prompt} processed.", model_name, latency, cost
    except Exception as e:
        return f"Error: {e}", None, None, None

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
            response, model_used, latency, cost = get_response(prompt)
            columns[i].write(f"Response from {model_used}:")
            columns[i].write(response)
            columns[i].write(f"Latency: {latency:.2f} seconds")
            columns[i].write(f"Cost: ${cost:.4f}")
        else:
            response, model_used, latency, cost = get_response_from_model(prompt, model)
            columns[i].write(f"Response from {model_used}:")
            columns[i].write(response)
            columns[i].write(f"Latency: {latency:.2f} seconds")
            columns[i].write(f"Cost: ${cost:.4f}")

# Optional: Display model details
if st.checkbox("Show Model Details"):
    st.write("Model Details:")
    for model, details in models.items():
        st.write(f"Model: {model}")
        st.write(f"Vendor: {details['vendor']}")
        st.write(f"Cost per Prompt: {details['cost_per_prompt']}")
        st.write(f"Cost per Completion: {details['cost_per_completion']}")
        st.write("----")
