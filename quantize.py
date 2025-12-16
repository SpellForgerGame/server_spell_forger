from model.matrix_regression_model import MatrixRegressionModel
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
from torch import nn
from data.data_definitions import SPELL_FIELDS, NUM_CLASSES_PER_FIELD
import sys
from preprocess.preprocess import preprocess_prompt
import asyncio
from train_regression_model import denormalize_spell_features 
import numpy as np

 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def load_model(path, output_shape=None, base_model_name='bert-base-multilingual-uncased', device='cpu'):

  
    model_absolute_path = os.path.join(SCRIPT_DIR, path)

    tokenizer = AutoTokenizer.from_pretrained(model_absolute_path)
    model = MatrixRegressionModel(base_model_name=base_model_name, output_shape=output_shape)
    model.load_state_dict(torch.load(os.path.join(model_absolute_path, 'model.pt'), map_location=device))
    return model.to(device), tokenizer


status_model, status_tokenizer = load_model('bert_matrix_model')
 
 
# Set to evaluation mode
status_model.eval()


quantized_model = torch.quantization.quantize_dynamic(
    status_model,
    # Specify the layers you want to quantize.
    # For a BERT-based model, Linear layers are the primary targets.
    # If you have custom layers, you might need to list them explicitly.
    # Listing nn.Linear will quantize all torch.nn.Linear layers in the model.
    {nn.Linear},
    dtype=torch.qint8  # 8-bit signed integer
)
torch.save(quantized_model.state_dict(), 'quantized_model_dynamic.pt')