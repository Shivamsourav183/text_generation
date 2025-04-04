import streamlit as st
from transformers import pipeline

# Load the text generation model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

text_generator = load_model()

# Streamlit UI
st.title("🤖 AI Text Generator")
st.write("Enter a prompt and generate AI-powered text!")

# Input prompt from the user
user_prompt = st.text_area("Enter a prompt:", "")

# Generation settings
max_length = st.slider("Max Length", min_value=20, max_value=200, value=50)
temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1)

# Generate text when button is clicked
if st.button("Generate Text"):
    if user_prompt.strip():
        with st.spinner("Generating text..."):
            output_text = text_generator(user_prompt, max_length=max_length, do_sample=True, temperature=temperature)
            st.subheader("Generated Text:")
            st.write(output_text[0]["generated_text"])
    else:
        st.warning("Please enter a prompt before generating.")
