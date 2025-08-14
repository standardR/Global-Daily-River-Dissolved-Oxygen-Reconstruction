import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from model_function.model_function import DOformer, evaluate
from model_utils.model_utils import process_dates, encode_cyclical_features, SafeScaler, prepare_data, \
    ImprovedTimeSeriesDataset

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# data path
data_path = r"./data/Train_data.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv(data_path)
df_orig = df.copy()
# feature selection
categorical_cols = [col for col in df.columns if col.startswith('LithologicalClasses')]
numeric_cols = [col for col in df.columns if col not in ["DO", 'Date', 'Date_orig', 'Latitude', 'Longitude',
                                                         'Date_sin', 'Date_cos', 'Lat_sin', 'Lat_cos', 'Lon_sin',
                                                         'Lon_cos'] + categorical_cols]

df = process_dates(df)
df_orig = process_dates(df_orig)
df = encode_cyclical_features(df)
# data split
split_date = pd.to_datetime("2016-12-31")
train_mask = df['Date'] < split_date
df_train_val = df[train_mask].reset_index(drop=True)
df_test = df[~train_mask].reset_index(drop=True)
n_train_val = len(df_train_val)
split_idx = int(n_train_val * 0.8)
df_train = df_train_val.iloc[:split_idx].reset_index(drop=True)
df_val = df_train_val.iloc[split_idx:].reset_index(drop=True)
total_len = len(df_train) + len(df_val) + len(df_test)
print("="*30)
print(f"Training data length: {len(df_train)}, Validation data length: {len(df_val)}, Test data length: {len(df_test)}")
print(f"Total data length: {total_len}, "
      f"Train : Validation : Test ratio = {len(df_train) / total_len:.2f} :"
      f" {len(df_val) / total_len:.2f} : {len(df_test) / total_len:.2f}")
print("="*30)

scaler = SafeScaler()
numeric_indices = list(range(len(numeric_cols)))
# prepare data
X_train, y_train = prepare_data(df_train, numeric_cols, categorical_cols)
X_val, y_val = prepare_data(df_val, numeric_cols, categorical_cols)
X_test, y_test = prepare_data(df_test, numeric_cols, categorical_cols)
# scale data
X_train = scaler.fit_transform(X_train, numeric_indices)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

window_size = 120
batch_size = 128

train_dataset = ImprovedTimeSeriesDataset(X_train, y_train, window_size=window_size, step_size=1)
val_dataset = ImprovedTimeSeriesDataset(X_val, y_val, window_size=window_size, step_size=1)
test_dataset = ImprovedTimeSeriesDataset(X_test, y_test, window_size=window_size, step_size=1)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
criterion = nn.HuberLoss(reduction="mean")
# load best model
best_epoch = 27
best_model_path = f"save_model/model_epoch{best_epoch}_{window_size}_{batch_size}_{0.243:.3f}.pth"

input_dim = X_train.shape[1]
model = DOformer(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1
    ).to(device)

model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.eval()
test_loss = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")

model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds).flatten()
all_targets = np.concatenate(all_targets).flatten()

# performance metrics
nse = 1 - np.sum((all_targets - all_preds) ** 2) / np.sum((all_targets - np.mean(all_targets)) ** 2)
mae = np.mean(np.abs(all_targets - all_preds))
mse = np.mean((all_targets - all_preds) ** 2)
rmse = np.sqrt(mse)
pbias = 100 * np.sum(all_preds - all_targets) / np.sum(all_targets)
pearson_corr = np.corrcoef(all_targets, all_preds)[0, 1]

print(f"NSE: {nse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"NSE: {nse:.4f}")
print(f"Pbias: {pbias:.4f}%")
print(f"Pearson: {pearson_corr:.4f}")

# Validation -- predicting on ungaged stations

# your independent validation data path
val_data_path = r"./your independent validation data path"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df_val_new = pd.read_csv(val_data_path)
df_val_new = process_dates(df_val_new)
df_val_new = encode_cyclical_features(df_val_new)

# prepare validation data
X_val_new, y_val_new = prepare_data(df_val_new, numeric_cols, categorical_cols)
X_val_new = scaler.transform(X_val_new)
val_new_dataset = ImprovedTimeSeriesDataset(X_val_new, y_val_new,
                                            window_size=window_size,
                                            step_size=1)
val_new_loader = DataLoader(val_new_dataset, batch_size=batch_size)

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in val_new_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds).flatten()
all_targets = np.concatenate(all_targets).flatten()
dates = df_val_new['Date'][window_size - 1:].reset_index(drop=True)
valid_mask = ~np.isnan(all_targets)
filtered_targets = all_targets[valid_mask]
filtered_preds = all_preds[valid_mask]

# performance metrics for validation
mae = np.mean(np.abs(filtered_targets - filtered_preds))
mse = np.mean((filtered_targets - filtered_preds) ** 2)
rmse = np.sqrt(mse)
nse = 1 - np.sum((filtered_targets - filtered_preds) ** 2) / np.sum((filtered_targets - np.mean(filtered_targets)) ** 2)
pbias = 100 * np.sum(all_preds - all_targets) / np.sum(all_targets)
pearson_corr = np.corrcoef(all_targets, all_preds)[0, 1]

print(f"NSE: {nse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Pbias: {pbias:.4f}%")
print(f"Pearson: {pearson_corr:.4f}")

filtered_dates = dates[valid_mask]
# Prepare DataFrame for validation results
start_date = '2001-01-01'
end_date = '2022-12-31'
results_df = pd.DataFrame({
    'Date': pd.to_datetime(filtered_dates),
    'Actual': filtered_targets,
    'Predicted': filtered_preds
})

all_preds_df = pd.DataFrame({
    "Date": pd.to_datetime(dates),
    "Predicted": all_preds
})
# save predictions to CSV
df_val_results = df_val_new.iloc[window_size - 1:].copy()
df_val_results['Prediction'] = all_preds
df_val_results['True'] = all_targets
df_val_results.to_csv('validation_predictions.csv', index=False)
