import torch
from torch.utils.data import DataLoader
from iTransformer import Model
from data_loader import ETThDataset
import matplotlib.pyplot as plt
import numpy as np


# ======================
# Config
# ======================
class Config:
    seq_len = 96
    pred_len = 24
    d_model = 512
    n_heads = 8
    e_layers = 2
    d_ff = 2048
    dropout = 0.1
    factor = 5
    activation = 'gelu'
    output_attention = False
    use_norm = True
    embed = 'fixed'
    freq = 'h'
    class_strategy = None


# ======================
# Data
# ======================
train_dataset = ETThDataset("ETTh1.csv", split='train')
val_dataset = ETThDataset("ETTh1.csv", split='val')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ======================
# Model
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = Config()
model = Model(configs).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()


# ======================
# Train Loop
# ======================
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_enc, x_mark_enc, y, y_mark in train_loader:
        x_enc = x_enc.to(device)
        y = y.to(device)

        # Noneをそのまま渡す
        pred = model(x_enc, x_mark_enc, y, y_mark)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.6f}")

    # ======================
    # Validation
    # ======================
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x_enc, x_mark_enc, y, y_mark in val_loader:
            x_enc = x_enc.to(device)
            y = y.to(device)

            pred = model(x_enc, x_mark_enc, y, y_mark)
            loss = criterion(pred, y)

            val_loss += loss.item()

    print(f"Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader):.6f}")

# ======================
# Prediction & Plot
# ======================

test_dataset = ETThDataset("ETTh1.csv", split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()

with torch.no_grad():
    x_enc, x_mark_enc, y_true, y_mark = next(iter(test_loader))

    x_enc = x_enc.to(device)

    pred = model(x_enc, x_mark_enc, y_true, y_mark)

    pred = pred.cpu().numpy()
    y_true = y_true.numpy()

# ======================
# Plot
# ======================

# 変数番号（0～6）
feature_idx = 0

plt.figure(figsize=(12, 6))

plt.plot(
    y_true[0, :, feature_idx],
    label='True'
)

plt.plot(
    pred[0, :, feature_idx],
    label='Prediction'
)

plt.title(f'ETTh1 Forecast - Variable {feature_idx}')
plt.xlabel('Time Step')
plt.ylabel('Value')

plt.legend()

plt.show()