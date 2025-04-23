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

SEED = 42
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)
PAD_TOKEN = '<|pad|>'
MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'
NEW_MODEL = 'Llama-3-8B-Instruct-Whatsapp-LM-FineTuner'

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    use_fast=True, 
    force_download=True
    )
tokenizer.add_special_tokens({'pad_token':PAD_TOKEN})
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map ='auto',
    torch_dtype=torch.bfloat16 ,
    force_download=True  # important
)
model.resize_token_embeddings(len(tokenizer))