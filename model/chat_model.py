import random
from textwrap import dedent
from typing import Dict, List


import numpy as np
import pandas as pd
import seaborn as sms
import torch
import peft
from sklearn.model_selection import train_test_split
from torch.utils.data import dataloader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import sys
import os

# Add llama.cpp folder to the sys.path
sys.path.append(os.path.abspath("../../llama.cpp"))

import llama_cpp

SEED = 42
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)
PAD_TOKEN = '<|pad|>'
MODEL_NAME = '/home/tommy/Github/Project LM/llama.cpp/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'
NEW_MODEL = 'Llama-3-8B-Instruct-Whatsapp-LM-FineTuned'# Initialize the model manually
model = llama_cpp.Llama(model_path=MODEL_NAME)

MODEL_FOR_TOKEN ='meta-llama/Meta-Llama-3-8B-Instruct'

"""# Load custom tokenizer for Llama 3 (assuming the tokenizer was also converted into GGUF format)
tokenizer = AutoTokenizer.from_pretrained(MODEL_FOR_TOKEN, use_fast=True, force_download=True)
tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
tokenizer.padding_side = 'right'

# Load the model (You may need to use a custom model loader, as shown in your prior code)
# Use the specific class for Llama from Hugging Face, or load it through GGUF-specific function
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Uses available hardware (GPU/CPU)
    torch_dtype=torch.bfloat16,  # Great for large models in FP16/32
    force_download=True
)

model.resize_token_embeddings(len(tokenizer))  # Adjust token embeddings size to include the new pad token"
"""
print("Model Path:", MODEL_NAME)
# Try using a context manager for the model

# Define the prompt
prompt = "Translate this text to French: 'Hello, how are you?'"

response = model.generate(prompt) # or use `list(response_generator)[0]` to get the first result
print(response)

# Manually free the model after usage
model.close()
