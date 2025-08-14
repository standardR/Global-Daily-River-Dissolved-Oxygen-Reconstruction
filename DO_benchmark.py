import random
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model_function.model_function import *
from model_utils.model_utils import *

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_path = r"./data/Train_data.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(data_path)
df_orig = df.copy()
categorical_cols = [col for col in df.columns if col.startswith('LithologicalClasses')]
numeric_cols = [col for col in df.columns if col not in ["DO", 'Date', 'Date_orig', 'Latitude', 'Longitude',
                                                         'Date_sin', 'Date_cos', 'Lat_sin', 'Lat_cos', 'Lon_sin',
                                                         'Lon_cos'
                                                         ] + categorical_cols]
df = process_dates(df)
df_orig = process_dates(df_orig)
df = encode_cyclical_features(df)

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

X_train, y_train = prepare_data(df_train, numeric_cols, categorical_cols)
X_val, y_val = prepare_data(df_val, numeric_cols, categorical_cols)
X_test, y_test = prepare_data(df_test, numeric_cols, categorical_cols)
X_train = scaler.fit_transform(X_train, numeric_indices)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

window_size = 120
batch_size = 128

# ------------------------- main function -------------------------
if __name__ == "__main__":

    train_dataset = ImprovedTimeSeriesDataset(X_train, y_train, window_size=window_size, step_size=1)
    val_dataset = ImprovedTimeSeriesDataset(X_val, y_val, window_size=window_size, step_size=1)
    test_dataset = ImprovedTimeSeriesDataset(X_test, y_test, window_size=window_size, step_size=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    input_dim = X_train.shape[1]
    # choose model
    model = DOLSTM(
        input_dim=input_dim,
        d_model=128,
        num_layers=6,
        dropout=0.3
    ).to(device)

    # model = DOGRU(
    #     input_dim=input_dim,
    #     d_model=128,
    #     num_layers=3,
    #     dropout=0.2
    # ).to(device)

    # model = DOBiLSTM(
    #     input_dim=input_dim,
    #     d_model=64,
    #     num_layers=4,
    #     dropout=0.1
    # ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.HuberLoss(reduction="mean")
    history, best_epoch = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=200)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
