import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def create_dataset(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(Y)


class ETThDataset(Dataset):
    def __init__(self, file_path, seq_len=96, pred_len=24, split='train'):
        df = pd.read_csv(file_path)

        # 日付削除（最初の列がdate）
        data = df.iloc[:, 1:].values.astype(np.float32)

        # 正規化（z-score）
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-5
        data = (data - self.mean) / self.std

        # データ分割（公式に近い比率）
        n = len(data)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)

        if split == 'train':
            data = data[:train_end]
        elif split == 'val':
            data = data[train_end:val_end]
        else:
            data = data[val_end:]

        self.X, self.Y = create_dataset(data, seq_len, pred_len)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)

        # ダミー（使わないけど必要）
        x_mark = torch.zeros_like(x)
        y_mark = torch.zeros_like(y)

        return x, x_mark, y, y_mark