import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error  # <--- 修改点1：引入 MAE
import matplotlib.pyplot as plt
import time


# Configurations
np.set_printoptions(suppress=True, precision=6)
pd.options.display.float_format = '{:.6f}'.format
try:
    torch.set_printoptions(precision=6, sci_mode=False)
except TypeError:
    try:
        torch.set_printoptions(precision=6)
    except Exception:
        pass


def np_arr_to_fixed_str(arr, precision=6):
    return np.array2string(np.asarray(arr), formatter={'float_kind': lambda x: f"{x:.{precision}f}"})


FILENAME = 'dataset.xlsx'
SHEET_NAME = 'data'

train_size = 40000
test_size = 4000
BATCH_SIZE = 128
NUM_EPOCHS = 500

# Learning Rate
LR_START = 1e-3
LR_END = 1e-5
GAMMA = (LR_END/LR_START)**(1/(NUM_EPOCHS-1))

MODEL_OUT = 'DNN_state.pth'

# ramdom seed 0 24 42 78 100
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Evaluation Metrics
def safe_mape(y_true, y_pred):
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0



# Data Loading
df = pd.read_excel(FILENAME, sheet_name=SHEET_NAME)

X_all = df[['dPL', 'Hg', 'Dg', 'Rg', 'Fg', 'Hw1', 'Hw2', 'Hw3', 'Dw1', 'Dw2', 'Dw3', 'Vw1', 'Vw2', 'Vw3', 'w10', 'w20', 'w30']].values.astype(np.float32)
y_all = df[['w1_final', 'w2_final', 'w3_final']].values.astype(np.float32)

X_train_arr = X_all[:train_size]
y_train_arr = y_all[:train_size]
X_test_arr = X_all[train_size:train_size + test_size]
y_test_arr = y_all[train_size:train_size + test_size]

print(f"train: {len(X_train_arr)}, test: {len(X_test_arr)}")

scaler_X = StandardScaler().fit(X_train_arr)
scaler_y = StandardScaler().fit(y_train_arr)

X_train_scaled = scaler_X.transform(X_train_arr)
X_test_scaled = scaler_X.transform(X_test_arr)

y_train_scaled = scaler_y.transform(y_train_arr)
y_test_scaled = scaler_y.transform(y_test_arr)

X_train = torch.tensor(X_train_scaled).float().to(device)
y_train = torch.tensor(y_train_scaled).float().to(device)
X_test = torch.tensor(X_test_scaled).float().to(device)
y_test = torch.tensor(y_test_scaled).float().to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)


# Model Definition
class DNN(nn.Module):
    def __init__(self, input_dim=17, hidden_dims=None, output_dim=3):
        super(DNN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [40, 40]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


model = DNN(input_dim=X_all.shape[1], output_dim=y_all.shape[1]).to(device)
# Note: The training loss function remains MSELoss because neural networks train better with squared error gradients,
# but the evaluation will use MAE.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR_START)

# Clean and precise scheduler using the calculated GAMMA
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

print(f"Using LR Start={LR_START:.6f}, LR End={LR_END:.6f}, Calculated Gamma={GAMMA:.6f}")

# Training Loop
train_losses = []
start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    # Train phase
    model.train()
    running = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()
        optimizer.step()
        running += loss.item() * bx.size(0)
    train_loss = running / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Learning rate update
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | LR: {current_lr:.8f}")

torch.save(model.state_dict(), MODEL_OUT)

end = time.time()
print(f"Training completed in {end - start:.4f} seconds.")

# Testing
model.load_state_dict(torch.load(MODEL_OUT, map_location=device, weights_only=True))
model.eval()

y_true_list, y_pred_list = [], []
with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx)
        y_true_list.append(by.cpu().numpy())
        y_pred_list.append(out.cpu().numpy())

y_true = np.vstack(y_true_list)
y_pred = np.vstack(y_pred_list)

y_true_real = scaler_y.inverse_transform(y_true)
y_pred_real = scaler_y.inverse_transform(y_pred)

print("\nPerformance Metrics per Wind Farm:")
for i in range(y_all.shape[1]):
    mae = mean_absolute_error(y_true_real[:, i], y_pred_real[:, i])
    mape = safe_mape(y_true_real[:, i], y_pred_real[:, i])
    print(f' Wind Farm {i + 1} - MAE: {mae:.6f}, MAPE: {mape:.4f}%')

mae_total = mean_absolute_error(y_true_real.flatten(), y_pred_real.flatten())
mape_total = safe_mape(y_true_real.flatten(), y_pred_real.flatten())
print(f'\nOverall Performance - MAE: {mae_total:.6f}, MAPE: {mape_total:.4f}%')

# Visualization
wind_farm_names = ['Wind Farm 1', 'Wind Farm 2', 'Wind Farm 3']
n_show = min(200, len(y_true_real))

for i in range(y_all.shape[1]):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_real[:n_show, i], label='True Value', marker='o', linestyle='-', markersize=4)
    plt.plot(y_pred_real[:n_show, i], label='Predicted Value', marker='x', linestyle='--', markersize=4)

    mae = mean_absolute_error(y_true_real[:, i], y_pred_real[:, i])
    mape = safe_mape(y_true_real[:, i], y_pred_real[:, i])

    plt.xlabel('Sample Index')
    plt.ylabel(f'w{i + 1}_final')
    plt.title(f'{wind_farm_names[i]} - True vs Pred (First {n_show} Samples)\nMAE: {mae:.6f}, MAPE: {mape:.4f}%')
    plt.legend()
    plt.grid(True)
    plt.show()

print(f"\nModel state dictionary saved to '{MODEL_OUT}'.")