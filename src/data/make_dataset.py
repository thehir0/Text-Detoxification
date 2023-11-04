import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class FilteredDataset(Dataset):
    def __init__(self, test):
        if test:
            detox_data = pd.read_csv('../data/interim/test_filtered.csv')
        else:
            detox_data = pd.read_csv('../data/interim/train_filtered.csv')

        X, y_true = detox_data['reference'].values, detox_data['translation'].values
        self.X = X
        self.y = y_true

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


