from utils import retrieve_model, fetch_data_split, compute_scores

def run_prediction():
    print("[PRED] Loading trained model...")
    model = retrieve_model("models/linear_regression_model.joblib")

    print("[PRED] Loading test data...")
    _, X_te, _, y_te = fetch_data_split()

    preds = model.predict(X_te)
    r2, mse = compute_scores(y_te, preds)

    print(f"[PRED] R2 Score: {r2:.4f}")
    print(f"[PRED] MSE: {mse:.4f}")
    print(f"[PRED] Sample predictions (first 5): {preds[:5]}")

if __name__ == "__main__":
    run_prediction()

