import random
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import optuna
from optuna import TrialPruned
from model_function.model_function import DOTransformer, train_model, evaluate
from model_utils.model_utils import process_dates, encode_cyclical_features, SafeScaler, prepare_data, \
    ImprovedTimeSeriesDataset

# 保持原始数据预处理部分不变
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_path = r"./data/Train353.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载和预处理
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


def objective(trial):
    # 超参数搜索空间
    window_size = trial.suggest_categorical("window_size", [60, 90, 120])
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # 确保nhead是d_model的因数
    possible_nheads = []
    for n in [4, 8, 16]:
        if d_model % n == 0:
            possible_nheads.append(n)
    if not possible_nheads:
        raise TrialPruned()
    nhead = trial.suggest_categorical("nhead", possible_nheads)

    num_layers = trial.suggest_int("num_layers", 3, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)

    # 重新创建数据集
    train_dataset = ImprovedTimeSeriesDataset(X_train, y_train, window_size=window_size, step_size=1)
    val_dataset = ImprovedTimeSeriesDataset(X_val, y_val, window_size=window_size, step_size=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = DOTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4,
                                                           threshold=1e-3)
    criterion = nn.HuberLoss(reduction="mean")

    # 训练模型并支持剪枝
    history = {'val_loss': []}
    best_val_loss = float('inf')
    for epoch in range(40):
        # 训练步骤
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

        # 验证步骤
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
        current_lr = optimizer.param_groups[0]['lr']  # 直接获取优化器中的学习率值
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={current_lr:.6f}")

        # 报告中间结果
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise TrialPruned()

        # 更新最佳损失
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(max_resource="auto"))
    study.optimize(objective, n_trials=30)

    # 输出最佳参数
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f" {key}: {value}")

    # 将最佳超参数保存到txt文件
    with open("best_params_353_50.txt", "w") as f:
        f.write(f"Best Value: {trial.value}\n")
        f.write("Best Params:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()

    # 使用最佳参数训练最终模型
    best_params = study.best_params
    window_size = best_params["window_size"]
    d_model = best_params["d_model"]
    nhead = best_params["nhead"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"]
    batch_size = best_params["batch_size"]

    # 创建最终数据集
    train_dataset = ImprovedTimeSeriesDataset(X_train, y_train, window_size=window_size, step_size=1)
    val_dataset = ImprovedTimeSeriesDataset(X_val, y_val, window_size=window_size, step_size=1)
    test_dataset = ImprovedTimeSeriesDataset(X_test, y_test, window_size=window_size, step_size=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化最终模型
    final_model = DOTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.HuberLoss(reduction="mean")

    # 训练最终模型
    final_history, best_epoch = train_model(final_model, train_loader, val_loader, optimizer, criterion, device,
                                            epochs=200)

    # 可视化训练过程
    plt.plot(final_history['train_loss'], label='Train Loss')
    plt.plot(final_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 测试集评估
    test_loss = evaluate(final_model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}")
