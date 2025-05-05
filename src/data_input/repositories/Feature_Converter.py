import pandas as pd
import torch


class FeatureConverter:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols

    def transform(self, df):
        """Convert DataFrame to clean float32 tensor"""
        # Convert each column to numeric
        numeric_df = df[self.feature_cols].apply(pd.to_numeric, errors='coerce')

        # Fill remaining NaNs
        numeric_df = numeric_df.fillna(0)

        return torch.tensor(numeric_df.values, dtype=torch.float32)