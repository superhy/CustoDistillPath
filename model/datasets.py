import torch
from torch.utils.data import Dataset

class FeatureClsDataset(Dataset):
    def __init__(self, features_paths, labels):
        """
        Args:
            features_paths: List of paths to pre-stored features from different foundation models
            labels: Ground truth labels
        """
        self.features_list = []
        for path in features_paths:
            self.features_list.append(torch.load(path))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Concatenate features from different models
        combined_features = torch.cat([features[idx] for features in self.features_list], dim=0)
        return combined_features, self.labels[idx]