import streamlit as st
import os
import openai
from routellm.controller import Controller
import time
from anthropic import Anthropic
import subprocess
import re

os.environ['LITELLM_LOG'] = 'DEBUG'

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
    "gpt-3.5-turbo": {"vendor": "openai", "cost_per_prompt": 5e-6, "cost_per_completion": 1.5e-5},
    "gpt-4o": {"vendor": "openai", "cost_per_prompt": 5e-6, "cost_per_completion": 1.5e-5},
    "RouteLLM Router (MF)": {"vendor": "routellm", "cost_per_prompt": 0, "cost_per_completion": 0},
    "RouteLLM Router (BERT)": {"vendor": "routellm", "cost_per_prompt": 0, "cost_per_completion": 0},
}

# Streamlit App
st.title("LLM Router Application")

@st.cache_resource
def init_controller():
    try:
        st.write(f"Initializing controller with default threshold")
        controller = Controller(
            routers=["mf", "bert"],
            strong_model="gpt-4o",
            weak_model="gpt-3.5-turbo",
            config={
                "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
                "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"}
            }
        )
        st.write("Controller initialized successfully")
        return controller
    except Exception as e:
        st.error(f"Error initializing controller: {str(e)}")
        return None


#strong_model_pct = st.slider("Percentage of strong model usage", 0.0, 1.0, 0.5, 0.01)
#threshold = 0.11593  # Default threshold
threshold = 0.9

if 'controller' not in st.session_state:
    st.session_state['controller'] = init_controller()

controller =  st.session_state.get('controller',None) 

# Function to calculate cost
def calculate_cost(model_name, input_tokens, output_tokens):
    if model_name == "gpt-4o":
        return (input_tokens * 5e-6) + (output_tokens * 1.5e-5)
    elif model_name == "gpt-3.5-turbo":
        return (input_tokens * 2.5e-7) + (output_tokens * 1.25e-6)
    elif model_name.startswith("RouteLLM Router"):
        strong_model_cost = (input_tokens * 5e-6) + (output_tokens * 1.5e-5)
        weak_model_cost = (input_tokens * 2.5e-7) + (output_tokens * 1.25e-6)
        return (strong_model_cost + weak_model_cost) / 2
    else:
        return 0

# Function to get a response from the RouteLLM router
def get_response(prompt, router):
    try:
        start_time = time.time()
        # Added a check to make sure controller is not None before attemping to use it.
        if controller is None:
            st.error("Controller not properly initialized, skipping RouteLLM")
            return "RouteLLM Unavailable", "N/A", 0, 0, 0, 0, "N/A"

        response = controller.chat.completions.create(
            model=f"router-{router}-0.11593",
            messages=[{"role": "user", "content": prompt}]
        )
        st.write("Full response:", response)
        end_time = time.time()
        latency = end_time - start_time
        input_tokens = len(prompt)
        output_tokens = len(response.choices[0].message.content)
        cost = calculate_cost(f"RouteLLM Router ({router.upper()})", input_tokens, output_tokens)
        selected_model = response.model
        return response.choices[0].message.content, f"RouteLLM Router ({router.upper()})", latency, cost, input_tokens, output_tokens, selected_model
    except Exception as e:
        st.error(f"RouteLLM Error: {str(e)}")
        return f"Error: {str(e)}", None, None, None, None, None, None

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
            input_tokens = len(prompt)
            output_tokens = len(response.choices[0].message.content)
            cost = calculate_cost(model_name, len(prompt), len(response.choices[0].message.content))
            return response.choices[0].message.content, model_name, latency, cost, input_tokens, output_tokens, None
        elif model_name == "gpt-3.5-turbo":
            response = openai.chat.completions.create(
                       model="gpt-3.5-turbo",
                       messages=[{"role": "user", "content": prompt}],
                       max_tokens=1024,
                       temperature=0.7,
            )
            end_time = time.time()
            latency = end_time - start_time
            input_tokens = len(prompt)
            output_tokens = len(response.choices[0].message.content)
            cost = calculate_cost(model_name, len(prompt), len(response.choices[0].message.content))
            return response.choices[0].message.content, model_name, latency, cost, input_tokens, output_tokens, None
        elif model_name.startswith("RouteLLM Router"):
            if controller is not None:  # Check if controller is initialized
                router = "mf" if model_name.endswith("(MF)") else "bert"
                return get_response(prompt, router)
            else:
                st.error("RouteLLM Controller failed to initialize. Cannot use RouteLLM Router.")
                return "RouteLLM unavailable", model_name, 0, 0, 0, 0, None
        else:
            end_time = time.time()
            latency = end_time - start_time
            cost = 0
            input_tokens = len(prompt)
            output_tokens = len(prompt)
            return f"Simulated response from {model_name}: {prompt} processed.", model_name, latency, cost, input_tokens, output_tokens, None
    except Exception as e:
        return f"Error: {e}", None, None, None, None, None, None

selected_models = st.multiselect("Select Models", list(models.keys()))

# Input prompt
prompt = st.text_area("Enter your prompt:", height=100)

# Button to generate response
if st.button("Get Response"):
    try:
        columns = st.columns(len(selected_models))

        for i, model in enumerate(selected_models):
            response, model_used, latency, cost, input_tokens, output_tokens, selected_model = get_response_from_model(prompt, model)
            if response is not None and model_used is not None:
                columns[i].write(f"Response from {model_used}:")
                columns[i].write(response)
                if latency is not None:
                    columns[i].write(f"Latency: {latency:.2f} seconds")
                if cost is not None:
                    columns[i].write(f"Cost: ${cost:.4f}")
                columns[i].write(f"Input Tokens: {input_tokens}")
                columns[i].write(f"Output Tokens: {output_tokens}")
                if model_used.startswith("RouteLLM Router"):
                    columns[i].write(f"Selected Model: {selected_model}")
                else:
                    columns[i].write(f"Selected Model: {model_used}")
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
st.write(f"Current threshold: 0.11593")

