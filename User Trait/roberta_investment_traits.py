import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaPreTrainedModel,
    TrainingArguments,
    Trainer,
    RobertaConfig
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#########################################
# 1. DATA LOADING & PREPROCESSING
#########################################

# Load the dataset
df = pd.read_csv('user_input_data.csv')

# Strip extra spaces from column names
df.columns = [col.strip() for col in df.columns]

# Define text and label columns
text_column = "user_input"
label_columns = ["invest_style", "risk_pref", "time_horizon", "fin_situation", "sector_choice"]

# Encode each label column with LabelEncoder
label_encoders = {}
num_labels_dict = {}
for col in label_columns:
    df[col] = df[col].str.strip()  # Remove extra spaces from values
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_encoders[col] = encoder
    num_labels_dict[col] = len(encoder.classes_)
    print(f"Column '{col}': {num_labels_dict[col]} classes -> {encoder.classes_}")

#########################################
# 2. DATASET CLASS
#########################################
# Custom dataset that returns tokenized text and each label as a separate key.
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class InvestmentDataset(Dataset):
    def __init__(self, texts, df_labels, label_columns, max_length=512):
        self.encodings = tokenizer(list(texts), truncation=True, padding='max_length', max_length=max_length)
        self.labels = df_labels.reset_index(drop=True)
        self.label_columns = label_columns

    def __getitem__(self, idx):
        # Get tokenized inputs
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add each label as a separate key
        for col in self.label_columns:
            item[col] = torch.tensor(self.labels.loc[idx, col], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Split dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[text_column], df[label_columns], test_size=0.2, random_state=42
)

train_dataset = InvestmentDataset(train_texts, train_labels, label_columns)
val_dataset = InvestmentDataset(val_texts, val_labels, label_columns)

#########################################
# 3. CUSTOM MULTI-TASK MODEL
#########################################
# Model with a shared RoBERTa backbone and a separate classification head for each label.
class MultiTaskRoberta(RobertaPreTrainedModel):
    def __init__(self, config, num_labels_dict):
        super().__init__(config)
        self.num_labels_dict = num_labels_dict
        self.roberta = RobertaModel(config)
        hidden_size = config.hidden_size

        # Create one classifier head per label column
        self.classifiers = nn.ModuleDict()
        for key, num_labels in num_labels_dict.items():
            self.classifiers[key] = nn.Linear(hidden_size, num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Extract custom label keys from kwargs
        labels = {}
        for key in self.num_labels_dict.keys():
            if key in kwargs:
                labels[key] = kwargs.pop(key)

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token representation

        logits = {}
        for key, classifier in self.classifiers.items():
            logits[key] = classifier(pooled_output)

        loss = None
        if labels:
            loss_fct = nn.CrossEntropyLoss()
            loss = 0
            for key in logits.keys():
                loss += loss_fct(logits[key], labels[key])
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# Initialize configuration and the custom model
config = RobertaConfig.from_pretrained('roberta-base')
model = MultiTaskRoberta.from_pretrained('roberta-base', config=config, num_labels_dict=num_labels_dict)
model.to(device)

#########################################
# 4. CUSTOM DATA COLLATOR
#########################################
# Custom collate function to ensure all keys (including label columns) are preserved.
def custom_data_collator(features):
    batch = {}
    # For every key in the first feature, stack all corresponding values from each feature
    for key in features[0].keys():
        batch[key] = torch.stack([feature[key] for feature in features])
    return batch

#########################################
# 5. TRAINING ARGUMENTS & METRICS
#########################################
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metrics = {}
    for col in label_columns:
        col_logits = logits[col]
        preds = np.argmax(col_logits, axis=1)
        true = labels[col]
        metrics[f"accuracy_{col}"] = accuracy_score(true, preds)
        metrics[f"f1_{col}"] = f1_score(true, preds, average="weighted", zero_division=0)
        metrics[f"precision_{col}"] = precision_score(true, preds, average="weighted", zero_division=0)
        metrics[f"recall_{col}"] = recall_score(true, preds, average="weighted", zero_division=0)
    return metrics

# Set remove_unused_columns to False to prevent Trainer from filtering out extra keys
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True if device.type == "cuda" else False,
    remove_unused_columns=False,  # <-- Prevent removal of custom label keys
)

#########################################
# 6. TRAINER SETUP
#########################################
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=custom_data_collator,  # Use custom collator to preserve all keys
)

#########################################
# 7. TRAINING / SAVING THE MODEL
#########################################
def train_model():
    model_path = 'multitask_roberta_investment_model.pt'
    if os.path.exists(model_path):
        print("Loading existing model...")
        global model
        model = torch.load(model_path, map_location=device)
        model.to(device)
        trainer.model = model
        print("Model loaded. Skipping training.")
    else:
        print("Training model...")
        trainer.train()
        torch.save(model, model_path)
        print("Model saved.")

#########################################
# 8. INFERENCE FUNCTION
#########################################
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = {}
    for col in label_columns:
        pred_class = torch.argmax(outputs["logits"][col], dim=1).cpu().numpy()[0]
        decoded_label = label_encoders[col].inverse_transform([pred_class])[0]
        predictions[col] = decoded_label
    return predictions

#########################################
# 9. EXAMPLE USAGE
#########################################
if __name__ == "__main__":
    train_model()
    test_input = (
        "I want to secure financial stability while having the flexibility to take moderate risks. "
        "I am looking for short-term investments in the healthcare sector to achieve prompt gains."
    )
    preds = predict(test_input)
    print("Predictions:")
    for task, label in preds.items():
        print(f"  {task}: {label}")
