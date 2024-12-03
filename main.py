from transformers import Trainer, TrainingArguments,AutoModelForSequenceClassification
from engine import train_dataset, eval_dataset
import torch
from model.model import model
from safetensors.torch import load_file
# weights_path = "results/checkpoint-22219/model.safetensors"
# state_dict = load_file(weights_path)  # Load the weights
# model.load_state_dict(state_dict)
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    num_train_epochs=150,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    metric_for_best_model="accuracy",
)

# Define metrics function
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=None,  # Not required for time-series data
    compute_metrics=compute_metrics,
    
)

# Train the model
trainer.train()
