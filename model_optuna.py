import random
import optuna
import torch.optim as optim
from optuna import TrialPruned
from torch.utils.data import DataLoader
from model_function.model_function import *
from model_utils.model_utils import *

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
categorical_cols = [col for col in df.columns if col.startswith('LithologicalClasses')]
numeric_cols = [col for col in df.columns if col not in ["DO", 'Date', 'Date_orig', 'Latitude', 'Longitude',
                                                         'Date_sin', 'Date_cos', 'Lat_sin', 'Lat_cos', 'Lon_sin',
                                                         'Lon_cos'] + categorical_cols]

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

scaler = SafeScaler()
numeric_indices = list(range(len(numeric_cols)))
X_train, y_train = prepare_data(df_train, numeric_cols, categorical_cols)
X_val, y_val = prepare_data(df_val, numeric_cols, categorical_cols)
X_test, y_test = prepare_data(df_test, numeric_cols, categorical_cols)
X_train = scaler.fit_transform(X_train, numeric_indices)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]
batch_size = 128


def objective(trial):
    # hyperparameter tuning
    window_size = trial.suggest_categorical("window_size", [60, 90, 120])
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 3, 8)
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])

    train_dataset = ImprovedTimeSeriesDataset(X_train, y_train, window_size=window_size, step_size=1)
    val_dataset = ImprovedTimeSeriesDataset(X_val, y_val, window_size=window_size, step_size=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DOformer(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4,
                                                           threshold=1e-3)
    criterion = nn.HuberLoss(reduction="mean")

    history = {'val_loss': []}
    best_val_loss = float('inf')
    for epoch in range(35):

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={current_lr:.6f}")

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f" {key}: {value}")

    with open("best_params_benchmark_GRU.txt", "w") as f:
        f.write(f"Best Value: {trial.value}\n")
        f.write("Best Params:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    best_params = study.best_params
    window_size = best_params["window_size"]
    d_model = best_params["d_model"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"]

    train_dataset = ImprovedTimeSeriesDataset(X_train, y_train, window_size=window_size, step_size=1)
    val_dataset = ImprovedTimeSeriesDataset(X_val, y_val, window_size=window_size, step_size=1)
    test_dataset = ImprovedTimeSeriesDataset(X_test, y_test, window_size=window_size, step_size=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    final_model = DOformer(
        input_dim=input_dim,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.HuberLoss(reduction="mean")
    final_history, best_epoch = train_model(final_model, train_loader, val_loader, optimizer, criterion, device,
                                            epochs=200)
    test_loss = evaluate(final_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
