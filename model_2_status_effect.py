import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel,Trainer, TrainingArguments
import torch.nn.functional as F
import torch.nn as nn

STATUSES = ["health", "speed", "defense", "mana", "stamina"]
TRIGGERS = ["player", "ally", "enemy", "all"]
NUM_OUTPUTS = len(STATUSES) * len(TRIGGERS)
NUM_CLASSES = 3  # -1, 0, 1
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Example spells with natural language descriptions
example_spells = [
    {
        "description": "Covers enemies in a weakening mist that slows them down and reduces their defense.",
        "modifiers": {
            ("speed", "enemy"): -1,
            ("defense", "enemy"): -1,
        }
    },
    {
        "description": "Grants your allies a divine blessing, improving their health and defense.",
        "modifiers": {
            ("health", "ally"): 1,
            ("defense", "ally"): 1,
            ("mana", "player"): 1,
        }
    },
]

class SpellMatrixDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = tokenizer(item["description"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Convert -1,0,1 to class labels 0,1,2
        labels = torch.tensor([val + 1 for val in item["labels"]], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
class SpellMatrixClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifiers = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, NUM_CLASSES) for _ in range(NUM_OUTPUTS)])

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token

        logits = torch.stack([head(cls_output) for head in self.classifiers], dim=1)  # shape: (batch, 20, 3)

        if labels is not None:
            loss = sum(F.cross_entropy(logits[:, i], labels[:, i]) for i in range(NUM_OUTPUTS)) / NUM_OUTPUTS
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# Prepare your data
train_data = [
    {
        "description": spell["description"],
        "labels": [spell["modifiers"].get((s, t), 0) for s in STATUSES for t in TRIGGERS]
    }
    for spell in example_spells  # reuse your list
]

train_dataset = SpellMatrixDataset(train_data)

# Initialize model
model = SpellMatrixClassifier()

# Trainer setup
training_args = TrainingArguments(
    output_dir="./spell_matrix_model",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    report_to="none",
    logging_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

trainer.save_model("./saved_models/spell_matrix_classifier_model")
tokenizer.save_pretrained("./saved_models/spell_matrix_classifier_model_tokenizer")
print("SpellMatrixClassifier salvo em ./saved_models/spell_matrix_classifier_model")
print("Tokenizer para SpellMatrixClassifier salvo em ./saved_models/spell_matrix_classifier_model_tokenizer")