import os
import warnings
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
    RobertaConfig,
    logging as hf_logging
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset

# Optional: Suppress transformers warnings if desired
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# Data loading & preprocessing
df = pd.read_csv('user_input_data.csv')
df.columns = [col.strip() for col in df.columns]
text_column = "user_input"
label_columns = ["invest_style", "risk_pref", "time_horizon", "fin_situation", "sector_choice"]

label_encoders = {}
num_labels_dict = {}
for col in label_columns:
    # Ensure the column is string type and remove extra spaces
    df[col] = df[col].astype(str).str.strip()
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_encoders[col] = encoder
    num_labels_dict[col] = len(encoder.classes_)
    print(f"Column '{col}': {num_labels_dict[col]} classes -> {encoder.classes_}")

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Custom Dataset with dynamic padding (padding will be handled in the collator)
class InvestmentDataset(Dataset):
    def __init__(self, texts, labels_df, label_columns, max_length=512):
        self.texts = list(texts)
        self.labels_df = labels_df.reset_index(drop=True)
        self.label_columns = label_columns
        # Tokenize texts without padding; dynamic padding is applied later.
        self.encodings = tokenizer(self.texts, truncation=True, padding=False, max_length=max_length)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add each label as a separate key in the dictionary.
        for col in self.label_columns:
            item[col] = torch.tensor(self.labels_df.loc[idx, col], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.texts)

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[text_column], df[label_columns], test_size=0.2, random_state=42
)
train_dataset = InvestmentDataset(train_texts, train_labels, label_columns)
val_dataset = InvestmentDataset(val_texts, val_labels, label_columns)

# Multi-task model definition
class MultiTaskRoberta(RobertaPreTrainedModel):
    def __init__(self, config, num_labels_dict):
        super().__init__(config)
        self.num_labels_dict = num_labels_dict
        self.roberta = RobertaModel(config)
        hidden_size = config.hidden_size
        # Create a classifier for each task using a ModuleDict.
        self.classifiers = nn.ModuleDict({
            col: nn.Linear(hidden_size, num_labels)
            for col, num_labels in num_labels_dict.items()
        })
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Extract labels from kwargs if provided.
        labels = {col: kwargs.pop(col) for col in self.num_labels_dict if col in kwargs}
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output for classification.
        pooled_output = outputs[1]
        logits = {col: self.classifiers[col](pooled_output) for col in self.num_labels_dict}
        loss = None
        if labels:
            loss_fct = nn.CrossEntropyLoss()
            # Sum the loss across all tasks; alternatively, average if desired.
            loss = sum(loss_fct(logits[col], labels[col]) for col in logits)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# Model configuration and instantiation
config = RobertaConfig.from_pretrained('roberta-base')
model = MultiTaskRoberta.from_pretrained('roberta-base', config=config, num_labels_dict=num_labels_dict)
model.to(device)
if hasattr(torch, "compile"):
    model = torch.compile(model)

# Custom data collator for dynamic padding
def custom_data_collator(features):
    token_keys = ['input_ids', 'attention_mask']
    token_features = [{k: f[k] for k in token_keys} for f in features]
    padded_tokens = tokenizer.pad(token_features, padding=True, return_tensors="pt")
    # Stack non-token fields (labels) into tensors.
    for key in features[0]:
        if key not in token_keys:
            padded_tokens[key] = torch.stack([f[key] for f in features])
    return padded_tokens

# Metrics computation function for multi-task evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metrics = {}
    for col in label_columns:
        true_labels = np.array(labels[col])
        pred_logits = logits[col]
        preds = np.argmax(pred_logits, axis=1)
        metrics[f"accuracy_{col}"] = accuracy_score(true_labels, preds)
        metrics[f"f1_{col}"] = f1_score(true_labels, preds, average="weighted", zero_division=0)
        metrics[f"precision_{col}"] = precision_score(true_labels, preds, average="weighted", zero_division=0)
        metrics[f"recall_{col}"] = recall_score(true_labels, preds, average="weighted", zero_division=0)
    return metrics

# Training arguments with dataloader_num_workers set to 0 to avoid recursion issues.
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Using the current parameter name
    save_strategy="epoch",
    fp16=True if device.type == "cuda" else False,
    remove_unused_columns=False,
    dataloader_num_workers=0,  # Disable multiprocessing to prevent recursion issues
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=custom_data_collator,
)

# Function to train or load an existing model using state_dict for saving/loading
def train_model():
    model_path = 'multitask_roberta_investment_model.pt'
    if os.path.exists(model_path):
        print("Loading existing model...")
        # Load the saved state dict and update the model.
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        trainer.model = model
        print("Model loaded. Skipping training.")
    else:
        print("Training model...")
        trainer.train()
        # Save only the state dict to avoid issues with full model serialization.
        torch.save(model.state_dict(), model_path)
        print("Model saved.")

# Function to perform prediction on a single text input
def predict(text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = {}
    for col in label_columns:
        pred_class = torch.argmax(outputs["logits"][col], dim=1).cpu().numpy()[0]
        decoded_label = label_encoders[col].inverse_transform([pred_class])[0]
        predictions[col] = decoded_label
    return predictions

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
