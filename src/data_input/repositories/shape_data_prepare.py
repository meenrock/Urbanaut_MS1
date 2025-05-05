import torch


def prepare_data(df, feature_cols, target_cols):
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    targets = torch.tensor(df[target_cols].values, dtype=torch.float32)
    return features, targets

