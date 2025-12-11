

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import json
from data.data_definitions import SPELL_FIELDS, NUM_CLASSES_PER_FIELD



number_of_simple_spell_features = len(SPELL_FIELDS)

class MatrixRegressionModel(nn.Module):
    def __init__(self, base_model_name='bert-base-uncased', output_shape=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Define the number of classes for the first categorical output
        # Assuming SPELL_FIELDS[0] is the categorical one
        self.num_categorical_classes = NUM_CLASSES_PER_FIELD[SPELL_FIELDS[0]]

        # Define the number of remaining regression outputs
        self.num_regression_outputs = len(SPELL_FIELDS) - 1

        # Linear layer for the first (categorical) output
        self.categorical_head = nn.Linear(self.hidden_size, self.num_categorical_classes)

        # Linear layer for the remaining (regression) outputs
        self.regression_head = nn.Linear(self.hidden_size, self.num_regression_outputs)

        # Define loss functions
        self.categorical_loss_fn = nn.CrossEntropyLoss()
        self.regression_loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]

        categorical_logits = self.categorical_head(cls_token)
        regression_predictions = self.regression_head(cls_token)

        # When labels are provided, calculate the combined loss
        if labels is not None:
            # Split labels: first value for categorical, rest for regression
            # Note: The categorical label needs to be a long tensor of class indices
            categorical_labels = labels[:, 0].long().squeeze()
            #categorical_labels = labels[:, 0].long()
            regression_labels = labels[:, 1:]

            # Calculate losses
            categorical_loss = self.categorical_loss_fn(categorical_logits, categorical_labels)
            regression_loss = self.regression_loss_fn(regression_predictions, regression_labels)

            # Combine losses (you can adjust weights if needed)
            total_loss = categorical_loss + regression_loss

            # Return the combined loss during training
            return total_loss, categorical_logits, regression_predictions # Return loss, logits, and predictions
        else:
            # During inference, just return the outputs
            return categorical_logits, regression_predictions

class MatrixRegressionDataset(Dataset):
    def __init__(self, texts, matrices, tokenizer, max_length=128):
        self.texts = texts
        self.matrices = matrices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # --- ADD THIS PRINT STATEMENT ---
        # Check the raw Python list before tensor conversion
        #print(f"DEBUG: Dataset matrices[{idx}] (Python list): {self.matrices[idx]}, Type: {type(self.matrices[idx])}, Length: {len(self.matrices[idx])}")
        # --- END PRINT ---

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.matrices[idx][0], dtype=torch.float) # <--- ADD [0] HERE
        }


def train_model(texts, matrices, model, tokenizer, device='cpu',
                batch_size=8, lr=2e-5, epochs=3):
    dataset = MatrixRegressionDataset(texts, matrices, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    epoch_losses = [] # List to store loss for each epoch
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            # The forward method now returns the total_loss as the first element
            loss, _, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(loader)
        epoch_losses.append(avg_epoch_loss) # Append the average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    # Log losses to a JSONL file
    log_file = 'losses_log.jsonl'
    log_entry = {
        'label': 'spell_status',
        'values': epoch_losses
    }
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')
    print(f"Epoch losses logged to {log_file}")


    return model

def save_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    # Save the state dict of the model
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    tokenizer.save_pretrained(path)

def load_model(path, base_model_name='bert-base-multilingual-uncased'):
     # Define the number of classes and regression outputs when loading
    num_categorical_classes = NUM_CLASSES_PER_FIELD[SPELL_FIELDS[0]]
    num_regression_outputs = len(SPELL_FIELDS) - 1

    model = MatrixRegressionModel(base_model_name=base_model_name,
                                  num_categorical_classes=num_categorical_classes, # These are handled internally now
                                  num_regression_outputs=num_regression_outputs) # These are handled internally now

    # Load the state dict
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


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
