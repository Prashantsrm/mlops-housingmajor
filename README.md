
# 🏡 MLOps California Housing Price Regression

Hi!  
This repository contains my end-to-end MLOps pipeline for the California Housing regression challenge. The goal was to build, test, containerize, quantize, and automate the lifecycle of a regression model, showing practical machine learning operations skills.

## 🚀 Project Overview

- **Dataset:** California housing (from sklearn)
- **Model:** Linear regression
- **MLOps Features:** Automated training, quantization, testing, Dockerization, and GitHub Actions CI/CD
- **Artifacts:** Model and quantized model files saved in a separate directory (`models/`)

## 🗂️ Files & Directory Structure

```
mlops-housingmajor/
├── .github/workflows/ci.yml  # CI/CD workflow
├── src/                      # Source code
│   ├── train.py              # Training script
│   ├── predict.py            # Prediction script
│   ├── quantize.py           # Manual model quantization & test
│   ├── utils.py              # Shared functions
│   └── __init__.py
├── models/                   # Model/param folder (created after training)
├── tests/
│   └── test_train.py         # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🛠️ How to Run This Project

### **1. Setup**

Clone this repo and install requirements:
```bash
pip install -r requirements.txt
```

### **2. Train the Model**

```bash
python src/train.py
```
- This trains a linear regression model and saves it to `models/linear_regression_model.joblib`.

### **3. Quantize the Model**

```bash
python src/quantize.py
```
- This manually compresses (quantizes) the model parameters and saves them to `models/quant_params.joblib`.

### **4. Run Prediction**

```bash
python src/predict.py
```
- Loads the trained model and outputs metrics and example predictions.

### **5. Run the Tests**

```bash
pytest
```

### **6. Use Docker**

Build and test the container (model & code run identically inside Docker!):
```bash
docker build -t mlops-housing .
docker run --rm mlops-housing
```

## ⚙️ CI/CD

- Automated workflow (`.github/workflows/ci.yml`) for:
  - Running unit tests
  - Training and quantizing models
  - Persisting model artifacts between jobs
  - Building and running the Docker image
- This ensures everything works not just locally, but in clean environments and repeatable setups.

## 📈 Results & Output Explanation

### **Training & Prediction Example Output:**

```
[INFO] Fetching California Housing data...
[RESULT] R2: 0.576
[RESULT] MSE: 0.556
[RESULT] Max Error: 9.88
[RESULT] Mean Error: 0.533
...
[PRED] R2 Score: 0.576
[PRED] MSE: 0.5559
[PRED] Sample predictions (first 5): [0.72 1.76 2.71 2.84 2.60]
```

### **Quantization Example Output:**

```
[QZ] Model size (orig): 0.68 KB
[QZ] Model size (quant): 0.34 KB
[QZ] Max coef error: 0.0044
[QZ] Bias error: 0.0000
[QZ] Quantization quality: poor (max diff: 7.46)
[QZ] Max Prediction Error (quant): 9.80
[QZ] Mean Prediction Error (quant): 6.33
```

### **Tests:**

```
tests/test_train.py ....    # 4 passed
```

## 🟦**Comparison Table: Original vs Quantized Model**

| Metric                  | Original Model         | Quantized Model         |
|-------------------------|-----------------------|------------------------|
| Model Size              | 0.68 KB               | 0.34 KB                |
| Max Coefficient Error   | –                     | 0.0044                 |
| Max Prediction Error    | 9.88                  | 9.80                   |
| Mean Prediction Error   | 0.53                  | 6.33                   |
| R² Score                | 0.576                 | –                      |
| MSE                     | 0.556                 | –                      |
| Quantization Quality    | –                     | poor (max diff: 7.46)  |

**Interpretation:**  
- The quantized model saves 50% of the size but comes with a higher mean prediction error due to reduced precision.  
- Maximum error on the test set is similar between the original and quantized models.
- Useful for deploying on edge devices where memory is more critical than precision.

## 📝 Reflections

- **What went well:** MLOps automation means every PR or push is validated in a real, clean environment—including model training, testing, and containerization.
- **What I learned:** Paying attention to correct Python paths, artifact sharing between CI jobs, and model versioning is crucial for robust deployments.
- **What’s next:** In a real deployment, you might evaluate other quantization strategies or track artifacts in something like MLflow for more complex pipelines.

## 👤 Author

Prashant Mishra  
August 2025


*Thanks for reviewing my project!*


