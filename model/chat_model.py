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

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.add_special_tokens({'pad_token':PAD_TOKEN})
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config = quantization_config,
    device_map ='auto'
)
model.resize_token_embeddings(len(tokenizer, pad_to_multiple_of=8))