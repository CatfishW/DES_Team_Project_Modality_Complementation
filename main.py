from transformers import Trainer, TrainingArguments,AutoModelForSequenceClassification
from engine import train_dataset, eval_dataset
import torch
from model.model import model
from safetensors.torch import load_file
# weights_path = "results/checkpoint-22219/model.safetensors"
# state_dict = load_file(weights_path)  # Load the weights
# model.load_state_dict(state_dict)
# Training arguments
class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,  # Enalbe shuffling here
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )
    def get_eval_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,  # Disable shuffling here
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
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
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=None,  # Not required for time-series data
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
