from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def train_random_forest_model(df: pd.DataFrame, features: list, params: dict) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier using log-return target labels.

    Args:
        df (pd.DataFrame): The input DataFrame with OHLCV and features.
        features (list): List of column names used as features.
        params (dict): RandomForestClassifier hyperparameters.

    Returns:
        RandomForestClassifier: A trained model.
    """
    df = df.copy()
    df['target'] = (np.log(df['Close'].shift(-1) / df['Close']) > 0).astype(int)
    df.dropna(subset=features + ['target'], inplace=True)

    X = df[features]
    y = df['target']

    model = RandomForestClassifier(**params, random_state=15)
    model.fit(X, y)

    return model
