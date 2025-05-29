import streamlit as st
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import accelerate

import logging

for name, l in logging.root.manager.loggerDict.items():
    if "streamlit" in name:
        l.disabled = True

model_name = "shaddie/rocketry_roqeto_model" #

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def query_model(prompt, max_new_tokens=84):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


st.title("Roqeto, the Rocketry Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Say something!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt) 
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get model response 
    roqeto_reply = query_model(prompt)

    response = f"Roqeto: {roqeto_reply}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
