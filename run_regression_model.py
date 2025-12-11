

from model.matrix_regression_model import MatrixRegressionModel
from transformers import AutoTokenizer, AutoModel
import torch
import os
import json
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


# model, tokenizer
def run_model():

    return load_model('bert_matrix_model')


'''
def predict(text, model, tokenizer, device='cpu'):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors individually
    with torch.no_grad():
        output = model(**inputs)
    return output.squeeze(0).cpu().numpy()

'''
def predict(text, model, tokenizer, device='cpu'):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # The forward method now returns categorical logits and regression predictions separately
        categorical_logits, regression_predictions = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Get predicted categorical class (index)
    predicted_categorical_class = torch.argmax(categorical_logits, dim=-1).cpu().numpy()

    # Get predicted regression values
    predicted_regression_values = regression_predictions.cpu().numpy()

    # Combine the predictions into a single numpy array to match the original output format
    # The categorical prediction is the first value
    predicted_output = np.concatenate((predicted_categorical_class.reshape(-1, 1), predicted_regression_values), axis=1)

    return predicted_output.squeeze(0) # Squeeze if you are only predicting for one text at a time


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")

    #print(torch.version.cuda)
    user_input = sys.argv[1]
    model, tokenizer = run_model()

    user_input = asyncio.run(preprocess_prompt(user_input))
    
    answer = predict(user_input, model, tokenizer) 
    answer = answer[0]  
    answer = denormalize_spell_features(answer) 
    answer_text = '[' 
    for i in range(len(SPELL_FIELDS)):
        answer_text += f'{answer[i]},' 
    answer_text = answer_text[:-1] + ']'
    print('{"result":'+answer_text + '}')  
else: 
    print('[run_regression_model imported]')  