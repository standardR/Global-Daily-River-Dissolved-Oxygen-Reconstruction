import os
import time
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

window_size = 120
batch_size = 128
seed = 42


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        return torch.sqrt(self.mse(outputs, targets) + 1e-10)


def compute_ig_feature_importance(model, dataloader, device, n_steps, feature_names):
    model.eval()
    ig = IntegratedGradients(model)

    total_attributions = torch.zeros(len(feature_names), device=device)
    total_delta = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        inputs.requires_grad = True

        baseline = torch.zeros_like(inputs)

        attributions, delta = ig.attribute(
            inputs=inputs,
            baselines=baseline,
            target=0,
            n_steps=n_steps,
            return_convergence_delta=True
        )

        assert inputs.ndim == 3, f"Expected 3D input, got {inputs.ndim}D"
        assert attributions.shape == inputs.shape, "Attributions shape mismatch"

        total_attributions += torch.sum(torch.abs(attributions), dim=(0, 1))
        total_delta += torch.sum(torch.abs(delta)).item()
        total_samples += inputs.size(0)

        del attributions, delta
        torch.cuda.empty_cache()

    global_importance = total_attributions / total_samples
    global_delta = total_delta / total_samples

    return global_importance.detach().cpu().numpy(), global_delta


class DOformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embed(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EarlyStopper:

    def __init__(self, patience=10, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def should_stop(self, loss):
        if loss < self.min_loss - self.min_delta:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=200):
    stopper = EarlyStopper()
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_epoch = -1

    os.makedirs("save_model", exist_ok=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4,
                                                           threshold=1e-3)
    log_file = "save_model/training_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Training Log\n============\n")
    print(f"Training Begin\n============")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()
        print(
            f"Epoch {epoch + 1}/{epochs} | Current LR: {current_lr} |"
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f} sec")

        scheduler.step(val_loss)

        save_path = \
            f"save_model/model_epoch{epoch + 1}_{window_size}_{batch_size}_{val_loss:.3f}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}")

        epoch_log = (
            f"Epoch {epoch + 1}/{epochs} | LR: {current_lr} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(epoch_log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

        if stopper.should_stop(val_loss):
            print("Early stopping triggered")
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            break

    print(f"Best model: epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}")
    print(f"Training End\n============")

    return history, best_epoch


class DOLSTM(nn.Module):
    def __init__(self, input_dim, d_model=128, num_layers=3, dropout=0.1):
        super(DOLSTM, self).__init__()

        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        out = self.output(x)
        return out


class DOGRU(nn.Module):
    def __init__(self, input_dim, d_model=128, num_layers=3, dropout=0.1):
        super(DOGRU, self).__init__()

        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        out = self.output(x)
        return out


class DOBiLSTM(nn.Module):
    def __init__(self, input_dim, d_model=128, num_layers=3, dropout=0.1):
        super(DOBiLSTM, self).__init__()

        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        self.output = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)

        forward_output = x[:, -1, :self.lstm.hidden_size]
        backward_output = x[:, 0, self.lstm.hidden_size:]
        x = torch.cat([forward_output, backward_output], dim=-1)
        out = self.output(x)
        return out
