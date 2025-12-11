

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import os
import numpy as np
import json
from torch.nn.utils.rnn import pad_sequence

MODEL_NAME = 'bert-base-multilingual-uncased'
# Assuming SPELL_FIELDS and NUM_CLASSES_PER_FIELD are defined as in your notebook

TARGET_TYPE = ['enemy', 'player', 'ally', 'all']

EFFECT_STATUS = ['health', 'speed', 'defense', 'mana']

NUMBER_OF_COMBINATIONS_OF_STATUS_EFFECT = len(TARGET_TYPE) * len(EFFECT_STATUS)

class StatusMatrixRegressionModel(nn.Module):
    def __init__(self, base_model_name='bert-base-uncased', output_shape=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.encoder.config.hidden_size

        #print('hidden size is ' , self.hidden_size)
        # Define the number of remaining regression outputs
        self.num_regression_outputs = (NUMBER_OF_COMBINATIONS_OF_STATUS_EFFECT)


        # Linear layer for the remaining (regression) outputs
        self.regression_head = nn.Linear(self.hidden_size, self.num_regression_outputs)

        # Define loss functions
        self.regression_loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]

        #categorical_logits = self.categorical_head(cls_token)
        regression_predictions = self.regression_head(cls_token)

        # When labels are provided, calculate the combined loss
        if labels is not None:
            # Split labels: first value for categorical, rest for regression
            # Note: The categorical label needs to be a long tensor of class indices
            #categorical_labels = labels[:, 0].long().squeeze()
            #categorical_labels = labels[:, 0].long()
            regression_labels = labels#labels[:, 1:]

            # Calculate losses
            #categorical_loss = self.categorical_loss_fn(categorical_logits, categorical_labels)
            regression_loss = self.regression_loss_fn(regression_predictions, regression_labels)

            # Combine losses (you can adjust weights if needed)
            #total_loss = categorical_loss + regression_loss

            # Return the combined loss during training
            return regression_loss, None, None
            #return total_loss, categorical_logits, regression_predictions # Return loss, logits, and predictions
        else:
            # During inference, just return the outputs
            return regression_predictions,None,None #categorical_logits, regression_predictions
            #return categorical_logits, regression_predictions

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
            'labels': torch.tensor(self.matrices[idx], dtype=torch.float)
        }
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.matrices[idx][0], dtype=torch.float)
        }


def train_model(texts, matrices, model, tokenizer, device='cuda',
                batch_size=8, lr=2e-5, epochs=3):
    dataset = MatrixRegressionDataset(texts, matrices, tokenizer)
    def custom_collate_fn(batch):
        # 'batch' is a list of dictionaries, where each dictionary is a sample
        # returned by MatrixRegressionDataset.__getitem__

        # Separate the components from each sample in the batch
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        # Pad input_ids and attention_masks to the maximum length within the current batch.
        # `pad_sequence` is ideal for this.
        # `batch_first=True` ensures the batch dimension comes first (batch_size, sequence_length).
        # `padding_value` for input_ids should be the tokenizer's pad_token_id.
        # `padding_value` for attention_mask should be 0 (indicating padding).
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        # For 'labels' (your flattened matrices), if they are *always* 16 items long,
        # you can simply stack them. If they *could* be variable length, you would
        # also use `pad_sequence` here. Based on our previous conversation, they are
        # consistently 16 items, so `torch.stack` is appropriate.
        stacked_labels = torch.stack(labels)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': stacked_labels
        }
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()

    epoch_losses = []

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

        #print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Log losses to a JSONL file
    log_file = 'losses_log.jsonl'
    log_entry = {
        'label': 'spell_status',
        'values': epoch_losses
    }
    log_file = os.path.join(GOOGLE_DRIVE_ROOT_PATH, log_file)

    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    else:
      with open(log_file, 'a', encoding='utf-8') as f:
          f.write(json.dumps(log_entry) + '\n')
    print(f"Epoch losses logged to {log_file}")

    return model

def save_model(model, tokenizer, path):

    #path = os.path.join(GOOGLE_DRIVE_ROOT_PATH, path)
    os.makedirs(path, exist_ok=True)
    # Save the state dict of the model
    torch.save(model.state_dict(), os.path.join(path, 'model_effects.pt'))
    tokenizer.save_pretrained(path)

def load_model(path, base_model_name='bert-base-multilingual-uncased'):
     # Define the number of classes and regression outputs when loading
    #num_categorical_classes = NUM_CLASSES_PER_FIELD[SPELL_FIELDS[0]]
    #num_regression_outputs = len(SPELL_FIELDS) - 1
 
    model = StatusMatrixRegressionModel(base_model_name=base_model_name)
    # Load the state dict
    model.load_state_dict(torch.load(os.path.join(path, 'model_effects.pt'), map_location=torch.device('cpu')))
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def predict_effects(text, model, tokenizer, device='cpu'):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # The forward method now returns categorical logits and regression predictions separately
        regression_predictions = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Get predicted categorical class (index)
    #predicted_categorical_class = torch.argmax(categorical_logits, dim=-1).cpu().numpy()

    # Get predicted regression values
    #predicted_regression_values = regression_predictions.cpu().numpy()

    # Combine the predictions into a single numpy array to match the original output format
    # The categorical prediction is the first value
    #predicted_output = np.concatenate((predicted_categorical_class.reshape(-1, 1), predicted_regression_values), axis=1)
    predicted_output = regression_predictions[0].cpu().numpy()
    return predicted_output.squeeze(0) # Squeeze if you are only predicting for one text at a time


# Example usage (assuming data loading and device setup are done as in your notebook):
# texts, matrices = load_data_from_json('final_data.jsonl')
# MODEL_NAME = 'bert-base-multilingual-uncased'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model (output_shape is not used in the modified __init__)
# model = MatrixRegressionModel(base_model_name=MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Train the model
# model = train_model(texts, matrices, model, tokenizer, device=device, epochs=20)

# Save the model
# save_model(model, tokenizer, './mixed_output_model_bert')

# Load the model
# loaded_model, loaded_tokenizer = load_model('./mixed_output_model_bert', base_model_name=MODEL_NAME)
# loaded_model.to(device)

# Make predictions
# test_text = "A fiery spell with high power."
# prediction = predict(test_text, loaded_model, loaded_tokenizer, device=device)
# print(f"Prediction: {prediction}") # This will show the predicted categorical index and regression values