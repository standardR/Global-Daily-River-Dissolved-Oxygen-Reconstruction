import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def process_dates(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(['Date', 'Latitude', 'Longitude', "Temperature_30d", "Temperature_2m_Min", "Temperature_2m"],
                        ascending=[True, False, False, False, False, False]
                        ).reset_index(drop=True)
    df['Date_orig'] = df['Date'].copy()
    return df


def encode_cyclical_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())
    df['Date_sin'] = np.sin(2 * np.pi * df['Date_ordinal'] / 365.25)
    df['Date_cos'] = np.cos(2 * np.pi * df['Date_ordinal'] / 365.25)
    df['Lat_sin'] = np.sin(np.radians(df['Latitude']))
    df['Lat_cos'] = np.cos(np.radians(df['Latitude']))
    df['Lon_sin'] = np.sin(np.radians(df['Longitude'] + 180))
    df['Lon_cos'] = np.cos(np.radians(df['Longitude'] + 180))
    df = df.drop(['Month', 'Day', 'Date_ordinal'], axis=1, errors='ignore')
    return df


class SafeScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_indices = None

    def fit_transform(self, x, numeric_indices):
        self.numeric_indices = numeric_indices
        if len(numeric_indices) > 0:
            x[:, numeric_indices] = self.scaler.fit_transform(x[:, numeric_indices])
        return x

    def transform(self, x):
        if len(self.numeric_indices) > 0:
            x[:, self.numeric_indices] = self.scaler.transform(x[:, self.numeric_indices])
        return x


def prepare_data(df, numeric_cols, categorical_cols):
    cols = numeric_cols + categorical_cols + ['Date_sin', 'Date_cos', 'Lat_sin', 'Lat_cos', 'Lon_sin', 'Lon_cos', 'DO']
    data = df[cols].values.astype(np.float32)
    return data[:, :-1], data[:, -1:]


class ImprovedTimeSeriesDataset(Dataset):
    def __init__(self, features, targets, dates=None, window_size=120, step_size=1):
        self.features = features
        self.targets = targets
        self.dates = dates
        self.window_size = window_size
        self.step_size = step_size

    def __len__(self):
        return (len(self.features) - self.window_size) // self.step_size + 1

    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        end_idx = start_idx + self.window_size
        x = self.features[start_idx:end_idx]
        y = self.targets[end_idx - 1]
        if self.dates is not None:
            return torch.FloatTensor(x), torch.FloatTensor(y), self.dates[end_idx - 1]
        return torch.FloatTensor(x), torch.FloatTensor(y)
