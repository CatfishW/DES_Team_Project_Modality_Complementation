import torch
from torch import nn
from transformers import PreTrainedModel, BertConfig, BertModel


# Define a custom model for time-series classification
class TimeSeriesTransformer(PreTrainedModel):
    def __init__(self, config, num_features, num_classes,no_self_attn=False):
        super().__init__(config)
        self.embedding = nn.Linear(num_features, config.hidden_size)
        self.input_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.transformer = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_features = num_features
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.no_self_attn = no_self_attn

    
    def forward(self,input_ids=None,labels=None):
        # x: [batch_size, seq_len, num_features]
        if input_ids.size(-1) != self.num_features:
            #pad
            input_ids = torch.cat([input_ids, torch.zeros(input_ids.size(0), input_ids.size(1), self.num_features - input_ids.size(-1)).to(input_ids.device)], dim=-1)
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        embeddings = self.input_proj(embeddings)  # [batch_size, seq_len, hidden_size]
        if self.no_self_attn:
            cls_output = embeddings[:, 0, :]
        else:
            attention_mask = torch.ones(embeddings.size()[:2]).to(input_ids.device)  # Mask if needed
            transformer_output = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask)
            cls_output = transformer_output.last_hidden_state[:, 0, :]  # Use CLS token output
        logits = self.classifier(self.dropout(cls_output))  # [batch_size, num_classes]
        if labels is not None:
            if self.num_classes == 1:
                loss = self.loss_bce(logits.view(-1), labels.view(-1).float())
                return {"loss": loss, "logits": logits}
            else:
                loss = self.loss_fn(logits, labels)
                return {"loss": loss, "logits": logits}
        return {"logits": logits}

# Configuration
num_features = 30  # Replace with the number of features in your dataset
num_classes = 6    # Adjust to your number of classes
config = BertConfig(hidden_size=128, num_attention_heads=4, num_hidden_layers=2, hidden_dropout_prob=0.1)

# Initialize model
model = TimeSeriesTransformer(config, num_features, num_classes,no_self_attn=True)

# # Dummy input for testing
# batch_size = 32
# seq_len = 50
# dummy_input = torch.rand(batch_size, seq_len, num_features)
# output = model(dummy_input)
# print(output.shape)  # [batch_size, num_classes]
