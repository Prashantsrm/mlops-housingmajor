import numpy as np
from src.utils import fetch_data_split, build_regressor, persist_model, compute_scores

def run_training():
    """Entry point for model training."""
    print("[INFO] Fetching California Housing data...")
    X_tr, X_te, y_tr, y_te = fetch_data_split()

    print("[INFO] Initializing regression model...")
    reg = build_regressor()

    print("[INFO] Fitting model to training data...")
    reg.fit(X_tr, y_tr)

    # Generate predictions on test set
    preds = reg.predict(X_te)

    # Evaluate performance
    r2_val, mse_val = compute_scores(y_te, preds)
    max_err = np.max(np.abs(y_te - preds))
    mean_err = np.mean(np.abs(y_te - preds))

    print(f"[RESULT] R2: {r2_val:.4f}")
    print(f"[RESULT] MSE: {mse_val:.4f}")
    print(f"[RESULT] Max Error: {max_err:.4f}")
    print(f"[RESULT] Mean Error: {mean_err:.4f}")

    # Save model inside models/ directory
    out_model = "models/linear_regression_model.joblib"
    persist_model(reg, out_model)
    print(f"[INFO] Model written to {out_model}")

    return reg, r2_val, mse_val

if __name__ == "__main__":
    run_training()

