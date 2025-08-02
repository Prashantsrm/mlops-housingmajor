import numpy as np
import joblib
import os
from utils import (
    compress_to_uint8, decompress_from_uint8,
    retrieve_model, fetch_data_split, compute_scores
)

def quantize_main():
    """Entry for quantization logic."""
    print("[QZ] Loading model artifact...")
    mdl = retrieve_model("models/linear_regression_model.joblib")

    # Extract coefficients and bias
    coeffs = mdl.coef_
    bias = mdl.intercept_

    print(f"[QZ] Coefficient shape: {coeffs.shape}")
    print(f"[QZ] Intercept: {bias:.8f}")
    print(f"[QZ] Coefficients: {coeffs}")

    # Save uncompressed params for reference
    params_raw = {'coef': coeffs, 'intercept': bias}
    os.makedirs("models", exist_ok=True)
    joblib.dump(params_raw, "models/unquant_params.joblib")

    # Quantize coefficients
    q_coef, min_c, max_c = compress_to_uint8(coeffs)
    # Quantize bias (handled as scalar or array)
    q_bias, min_b, max_b = compress_to_uint8(np.array([bias]))

    print(f"\n[QZ] Quantizing bias (8-bit, per-coefficient)...")
    print(f"[QZ] Bias value: {bias:.8f}")

    # Handle bias as scalar or array safely
    if np.isscalar(q_bias) or np.shape(q_bias) == ():
        quant_intercept8 = q_bias
        int8_min = min_b
        int8_max = max_b
    else:
        quant_intercept8 = q_bias[0]
        int8_min = min_b[0]
        int8_max = max_b[0]

    params_quant = {
        'quant_coef8': q_coef,
        'coef8_min': min_c,
        'coef8_max': max_c,
        'quant_intercept8': quant_intercept8,
        'int8_min': int8_min,
        'int8_max': int8_max
    }
    joblib.dump(params_quant, "models/quant_params.joblib", compress=3)
    print("[QZ] Quantized params saved to models/quant_params.joblib")

    sz_quant = os.path.getsize("models/quant_params.joblib")
    sz_orig = os.path.getsize("models/linear_regression_model.joblib")
    print(f"[QZ] Model size (orig): {sz_orig / 1024:.2f} KB")
    print(f"[QZ] Model size (quant): {sz_quant / 1024:.2f} KB")
    print(f"[QZ] Size delta: {(sz_orig - sz_quant) / 1024:.2f} KB")

    # Dequantize for test
    d_coef = decompress_from_uint8(q_coef, min_c, max_c)
    d_bias = decompress_from_uint8(np.array([quant_intercept8]), np.array([int8_min]), np.array([int8_max]))

    # Error checks
    err_coef = np.abs(coeffs - d_coef).max()
    err_bias = np.abs(bias - d_bias)
    print(f"[QZ] Max coef error: {err_coef:.8f}")
    bias_error_scalar = float(err_bias) if isinstance(err_bias, np.ndarray) else err_bias
    print(f"[QZ] Bias error: {bias_error_scalar:.8f}")


    # Inference test
    X_tr, X_te, y_tr, y_te = fetch_data_split()
    pred_manual = X_te[:5] @ d_coef + d_bias

    print("\n[QZ] Inference check (first 5):")
    print(f"[QZ] Manual dequant preds: {pred_manual}")

    diff = np.abs(mdl.predict(X_te[:5]) - pred_manual)
    print(f"[QZ] Abs diff: {diff}")
    print(f"[QZ] Max diff: {diff.max()}")
    print(f"[QZ] Mean diff: {diff.mean():.6f}")
    if diff.max() < 0.1:
        print(f"[QZ] Quantization quality: good (max diff: {diff.max():.6f})")
    elif diff.max() < 1.0:
        print(f"[QZ] Quantization quality: ok (max diff: {diff.max():.6f})")
    else:
        print(f"[QZ] Quantization quality: poor (max diff: {diff.max():.6f})")

    # Prediction errors for quantized
    max_pred_err = np.max(np.abs(y_te - (X_te @ d_coef + d_bias)))
    mean_pred_err = np.mean(np.abs(y_te - (X_te @ d_coef + d_bias)))
    print(f"[QZ] Max Prediction Error (quant): {max_pred_err:.4f}")
    print(f"[QZ] Mean Prediction Error (quant): {mean_pred_err:.4f}")
    print("\n[QZ] Quantization done!\n")

    # R2/MSE for quantized
    y_pred_quant = X_te @ d_coef + d_bias
    r2_quant, mse_quant = compute_scores(y_te, y_pred_quant)
    print(f"[QZ] R2 (quant): {r2_quant:.4f}")
    print(f"[QZ] MSE (quant): {mse_quant:.4f}")

if __name__ == "__main__":
    quantize_main()

