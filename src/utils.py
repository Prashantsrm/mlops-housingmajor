import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def fetch_data_split(test_size=0.2, random_state=42):
    """Load California housing data and split into train/test sets."""
    X, y = fetch_california_housing(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_regressor():
    """Return a new LinearRegression model instance."""
    return LinearRegression()

def persist_model(model, filepath):
    """Save the sklearn model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def retrieve_model(filepath):
    """Load model or params from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)

def compute_scores(y_true, y_pred):
    """Compute R2 and MSE metrics."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse

def compress_to_uint8(arr: np.ndarray):
    """
    Compress float array to uint8 by linear scaling.
    Returns quantized uint8 array, min, max for dequantization.
    """
    arr_min = arr.min() if hasattr(arr, 'min') else arr
    arr_max = arr.max() if hasattr(arr, 'max') else arr
    if arr_max == arr_min:
        # Range zero, all values are same
        # Return zero array or scalar zero quant
        if arr.size == 1:
            return np.uint8(0), arr_min, arr_max
        else:
            return np.zeros_like(arr, dtype=np.uint8), arr_min, arr_max
    scale = 255 / (arr_max - arr_min)
    arr_q = ((arr - arr_min) * scale).astype(np.uint8)
    return arr_q, arr_min, arr_max


def decompress_from_uint8(arr_q: np.ndarray, arr_min: float, arr_max: float):
    """
    Decompress uint8 array back to float using min/max.
    """
    scale = 255 / (arr_max - arr_min)
    return arr_q.astype(np.float32) / scale + arr_min

