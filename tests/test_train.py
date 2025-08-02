import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from sklearn.linear_model import LinearRegression
from src.utils import fetch_data_split, build_regressor

def test_dataset_loads():
    X_tr, X_te, y_tr, y_te = fetch_data_split()
    assert X_tr.shape[0] > 0
    assert X_te.shape[0] > 0
    assert y_tr.shape[0] > 0
    assert y_te.shape[0] > 0

def test_model_instance():
    reg = build_regressor()
    assert isinstance(reg, LinearRegression)

def test_model_training():
    X_tr, _, y_tr, _ = fetch_data_split()
    reg = build_regressor()
    reg.fit(X_tr, y_tr)
    assert hasattr(reg, "coef_")
    assert reg.coef_.shape[0] == X_tr.shape[1]

def test_r2_threshold():
    X_tr, X_te, y_tr, y_te = fetch_data_split()
    reg = build_regressor()
    reg.fit(X_tr, y_tr)
    r2 = reg.score(X_te, y_te)
    assert r2 > 0.5, f"R2 score should be > 0.5, got {r2:.4f}"

