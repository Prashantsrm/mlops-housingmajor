
# 🏡 MLOps California Housing Price Regression

Welcome!  
This repository provides a complete MLOps pipeline for the California Housing regression problem. From data loading, through training, manual quantization, prediction, testing, containerization, and CI/CD automation — this project shows practical skills in machine learning operations.

## 🚀 Project Summary

- **Dataset:** California Housing from `sklearn`
- **Model:** Linear Regression
- **Quantization:** Manual 8-bit quantization of model parameters to reduce size
- **Testing:** Unit tests with `pytest`
- **Containerization:** Docker for reproducibility
- **CI/CD:** Automated pipeline with GitHub Actions
- **Model artifacts:** Stored in `models/` directory, passed between CI jobs as artifacts

## 📁 Structure Overview

```
mlops-housingmajor/
├── .github/workflows/ci.yml    # GitHub Actions workflow for CI/CD
├── src/
│   ├── train.py                # Trains model, saves to models/
│   ├── quantize.py             # Performs manual quantization, saves quantized params
│   ├── predict.py              # Loads model and runs predictions
│   ├── utils.py                # Reusable helper functions
│   └── __init__.py             # Package marker
├── models/                     # Stores model artifacts (not git-tracked)
├── tests/
│   └── test_train.py           # Basic tests on data loading and training
├── Dockerfile                  # Docker container definition
├── requirements.txt
└── README.md                   # This file
```

## 🛠 Running Instructions

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py
```

- Outputs `models/linear_regression_model.joblib`.

### Quantize the model

```bash
python src/quantize.py
```

- Outputs quantized parameters to `models/quant_params.joblib`.
- Prints size reduction and quantization quality info.

### Predict using the model

```bash
python src/predict.py
```

- Displays R², MSE, and sample predictions on the test set.

### Run unit tests

```bash
pytest
```

- Runs tests verifying data loading, model initialization, and training steps.

### Build and Run Docker container

```bash
docker build -t mlops-housing .
docker run --rm mlops-housing
```

- Containerizes the entire pipeline for consistent execution.

## 📊 Output Summary

### Training output:

```
[RESULT] R2: 0.5758
[RESULT] MSE: 0.5559
[RESULT] Max Error: 9.8753
[RESULT] Mean Error: 0.5332
Model written to models/linear_regression_model.joblib
```

### Predict output:

```
R2 Score: 0.5758
MSE: 0.5559
Sample predictions (first 5): [0.7191 1.764 2.710 2.838 2.605]
```

### Quantization output:

```
Model size (orig): 0.68 KB
Model size (quant): 0.34 KB
Size delta: 0.34 KB
Max coef error: 0.0044
Bias error: 0.0000
Quantization quality: poor (max diff: 7.463)
Max Prediction Error (quant): 68.894
Mean Prediction Error (quant): 6.330
R2 (quant): -46.684
MSE (quant): 62.485
```

### Unit tests:

```
4 passed in 3.33s
```

## 📈 Results Comparison

| Metric                      | Original Model   | Quantized Model       |
|-----------------------------|------------------|----------------------|
| Model Size                  | 0.68 KB          | 0.34 KB              |
| Max Coefficient Error       | –                | 0.0044               |
| Max Prediction Error        | 9.88             | 68.89                |
| Mean Prediction Error       | 0.53             | 6.33                 |
| R² Score                   | 0.576            | -46.68               |
| MSE                        | 0.56             | 62.48                |
| Quantization Quality        | good             | poor (max diff: 7.46)|

## 📋 Analysis

- The **8-bit manual quantization** successfully halves the model size.
- However, this aggressive quantization causes a **large degradation in prediction accuracy**, evident in the negative R² and much higher error metrics.
- Regression models are sensitive to parameter perturbations; naive quantization leads to large deviations in output.
- This tradeoff between model size and accuracy is crucial in MLOps pipelines.
- More refined quantization techniques (e.g., quantization-aware training, mixed precision) would be necessary for production-quality compressed regression models.

## 🔄 Workflow Automation

- The entire pipeline (train → quantize → test → docker build) is automated using GitHub Actions.
- Artifacts are passed between jobs securely.
- Docker ensures consistent runtime environments on local machines and CI servers.

## 🙌 Author

Prashant Kumar Mishra  
Roll No.- G24AI1103

Thank you for reviewing this project!  
