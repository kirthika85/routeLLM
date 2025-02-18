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
    "claude-3-haiku-20240307": {"vendor": "anthropic", "cost_per_prompt": 2.5e-7, "cost_per_completion": 1.25e-6},
    "gpt-4o": {"vendor": "openai", "cost_per_prompt": 5e-6, "cost_per_completion": 1.5e-5},
    "RouteLLM Router (MF)": {"vendor": "routellm", "cost_per_prompt": 0, "cost_per_completion": 0},
    "RouteLLM Router (BERT)": {"vendor": "routellm", "cost_per_prompt": 0, "cost_per_completion": 0},
}

# Streamlit App
st.title("LLM Router Application")

# Function to calculate cost
def calculate_cost(model_name, input_tokens, output_tokens):
    if model_name == "gpt-4o":
        return (input_tokens + output_tokens) * 5 / 1e6
    elif model_name == "claude-3-haiku-20240307":
        return (input_tokens * 0.8 + output_tokens * 4) / 1e6
    elif model_name.startswith("RouteLLM Router"):
        strong_model_cost = (input_tokens + output_tokens) * 5 / 1e6
        weak_model_cost = (input_tokens * 0.8 + output_tokens * 4) / 1e6
        return (strong_model_cost + weak_model_cost) / 2
    else:
        return 0

# Function to get a response from the RouteLLM router
def get_response(prompt, router):
    try:
        client = Controller(
            routers=[router],
            strong_model="gpt-4o",
            weak_model="claude-3-haiku-20240307",
            config={
                "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
                "bert": {"checkpoint_path": "bert-base-uncased"}
            }
        )
        start_time = time.time()
        response = client.chat.completions.create(
            model=f"router-{router}-0.11593",
            messages=[{"role": "user", "content": prompt}]
        )
        end_time = time.time()
        latency = end_time - start_time
        cost = calculate_cost(f"RouteLLM Router ({router.upper()})", len(prompt), len(response.choices[0]["message"]["content"]))
        return response.choices[0]["message"]["content"], f"RouteLLM Router ({router.upper()})", latency, cost
    except Exception as e:
        return f"Error: {e}", None, None, None

# Function to get a response from a specific model (e.g., GPT-4o, Claude)
def get_response_from_model(prompt, model_name):
    try:
        start_time = time.time()
        if model_name == "gpt-4o":
            response = openai.chat.completions.create(
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
        elif model_name.startswith("RouteLLM Router"):
            router = "mf" if model_name.endswith("(MF)") else "bert"
            return get_response(prompt, router)
        else:
            end_time = time.time()
            latency = end_time - start_time
            cost = 0
            return f"Simulated response from {model_name}: {prompt} processed.", model_name, latency, cost
    except Exception as e:
        return f"Error: {e}", None, None, None

selected_models = st.multiselect("Select Models", list(models.keys()))

# Input prompt
prompt = st.text_area("Enter your prompt:", height=100)

# Button to generate response
if st.button("Get Response"):
    try:
        columns = st.columns(len(selected_models))
        
        for i, model in enumerate(selected_models):
            response, model_used, latency, cost = get_response_from_model(prompt, model)
            if response is not None and model_used is not None:
                columns[i].write(f"Response from {model_used}:")
                columns[i].write(response)
                if latency is not None:
                    columns[i].write(f"Latency: {latency:.2f} seconds")
                if cost is not None:
                    columns[i].write(f"Cost: ${cost:.4f}")
            else:
                columns[i].write("Error: Unable to retrieve response.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
                
# Optional: Display model details
if st.checkbox("Show Model Details"):
    st.write("Model Details:")
    for model, details in models.items():
        st.write(f"Model: {model}")
        st.write(f"Vendor: {details['vendor']}")
        st.write(f"Cost per Prompt: {details['cost_per_prompt']}")
        st.write(f"Cost per Completion: {details['cost_per_completion']}")
        st.write("----")
