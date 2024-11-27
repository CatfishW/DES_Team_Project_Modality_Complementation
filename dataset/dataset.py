from torch.utils.data import Dataset
import torch
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels, num_features_to_drop=None):
        self.features = features  # Shape: [num_samples, seq_len, num_features]
        self.labels = labels      # Shape: [num_samples]
        self.num_features_to_drop = num_features_to_drop

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.num_features_to_drop is None:
            return {"input_ids": torch.tensor(self.features[idx], dtype=torch.float32),
                    "labels": torch.tensor(self.labels[idx], dtype=torch.long)}
        else:
            feature_sample = self.features[idx]
            num_features = feature_sample.shape[1]
            features_to_keep = np.random.choice(num_features, num_features - self.num_features_to_drop, replace=False)
            reduced_features = feature_sample[:, features_to_keep]
            return {"input_ids": torch.tensor(reduced_features, dtype=torch.float32),
                    "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

# Example usage
if __name__ == "__main__":
    # Example dataset with shape (50, 10, 14)
    data = np.random.rand(50, 10, 14)
    labels = np.random.randint(0, 2, size=(50,))

    # Create dataset with 2 features to drop
    dataset = TimeSeriesDataset(data, labels, num_features_to_drop=2)
    print(len(dataset))  # Should print 50
    print(dataset[0])  # Should print the first sample with reduced features