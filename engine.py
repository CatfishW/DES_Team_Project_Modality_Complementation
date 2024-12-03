from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM
from config.config import cfg
from model.model import model
from dataset.dataset import TimeSeriesDataset
#SET CUDA AVAILABLE DEVICES
import tqdm
import os
import pandas as pd
# Load CSV data
file_path = "./dataset/OD_train.csv"
data = pd.read_csv(file_path)
# Display the first few rows
print(data.head())
import numpy as np

# Separate features and labels
features = data.drop(columns=["fatigue","F","block","time"]).values  # Replace "fatigue" with your label column
labels = data["fatigue"].values  # Target labels

# Optionally normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Reshape for sequence format (if needed)
# Example: Split into sequences with shape [num_samples, seq_len, num_features]
seq_len = 50  # Adjust this based on your needs
num_features = features.shape[1]

# Create sequences
sequences = []
sequence_labels = []

for i in range(len(features) - seq_len + 1):
    sequences.append(features[i : i + seq_len])
    sequence_labels.append(labels[i + seq_len - 1])

sequences = np.array(sequences)  # Shape: [num_sequences, seq_len, num_features]
sequence_labels = np.array(sequence_labels)  # Shape: [num_sequences]
from sklearn.model_selection import train_test_split

train_features, eval_features, train_labels, eval_labels = train_test_split(
    sequences, sequence_labels, test_size=0.2, random_state=42
)

# Print dataset shapes for verification
print("Train Features Shape:", train_features.shape)
print("Train Labels Shape:", train_labels.shape)
print("Eval Features Shape:", eval_features.shape)
print("Eval Labels Shape:", eval_labels.shape)
from torch.utils.data import DataLoader

# Define train and eval datasets
train_dataset = TimeSeriesDataset(train_features, train_labels,13)
eval_dataset = TimeSeriesDataset(eval_features, eval_labels,13)

# Optional: Create DataLoaders for manual training (if not using Trainer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32)





# os.environ["CUDA_VISIBLE_DEVICES"] = cfg['model']['CUDA_VISIBLE_DEVICES']
# if cfg['model']['train_loop_type'] == "huggingface":
#     training_args = TrainingArguments(
#         output_dir=cfg['model']['output_dir'],
#         num_train_epochs=cfg['model']['num_train_epochs'],
#         per_device_train_batch_size=cfg['model']['per_device_train_batch_size'],
#         per_device_eval_batch_size=cfg['model']['per_device_eval_batch_size'],
#         warmup_steps=cfg['model']['warmup_steps'],
#         weight_decay=cfg['model']['weight_decay'],
#         logging_dir=cfg['model']['logging_dir'],
#         logging_steps=cfg['model']['logging_steps'],
#         save_steps=cfg['model']['save_steps'],
#         evaluation_strategy=cfg['model']['evaluation_strategy'],
#         eval_steps=cfg['model']['eval_steps'],
#         save_total_limit=cfg['model']['save_total_limit'],
#         load_best_model_at_end=cfg['model']['load_best_model_at_end'],
#         metric_for_best_model=cfg['model']['metric_for_best_model'],
#         greater_is_better=cfg['model']['greater_is_better'],
#         report_to=cfg['model']['report_to'],
#         run_name=cfg['model']['run_name'],
#         seed=cfg['model']['seed'],
#         disable_tqdm=cfg['model']['disable_tqdm']
#     )
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=cfg['model']['train_dataset'],
#         eval_dataset=cfg['model']['eval_dataset'],
#         compute_metrics=compute_metrics
#     )
# else:
#     #add text of to be implemented on Error
#     raise NotImplementedError("Only huggingface train loop is implemented for now.")