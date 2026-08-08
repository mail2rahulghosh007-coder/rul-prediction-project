# Turbofan Engine Remaining Useful Life (RUL) Prediction

Predicting how many operational cycles remain before a jet engine needs maintenance, using deep learning on multivariate sensor time-series data — benchmarked against classical ML, tracked with MLflow, versioned with DVC, and deployed as a live web app.

## 🔗 Project Links & Live Demo

* **GitHub Repository:** [github.com/mail2rahulghosh007-coder/rul-prediction-project](https://github.com/mail2rahulghosh007-coder/rul-prediction-project)
* **Live App Demo:** [rul-prediction-project-curwtuldsgqrwxqdgapjrg.streamlit.app](https://rul-prediction-project-curwtuldsgqrwxqdgapjrg.streamlit.app)
* **MLOps Tracking (DagsHub):** [dagshub.com/mail2rahulghosh007-coder/rul_prediction_project](https://dagshub.com/mail2rahulghosh007-coder/rul_prediction_project)

---

## Problem

Predictive maintenance is a high-value, real-world application of deep learning across manufacturing, aviation, and energy. Given a stream of sensor readings from a jet engine over its operational life, the goal is to predict **Remaining Useful Life (RUL)** — how many more cycles the engine can run before failure — so maintenance can be scheduled proactively instead of reactively.

This project uses NASA's **C-MAPSS FD001** dataset: simulated turbofan engine degradation data with 21 sensors and 3 operational settings recorded per cycle, across 100 engines run to failure.

## Approach

Rather than committing to a single architecture, this project treats model selection as an empirical question: three approaches were built, fairly benchmarked on identical data splits, and compared — deep learning isn't assumed to win by default.

| Model | Test RMSE | Test MAE |
|---|---|---|
| **CNN-LSTM** (final choice) | **13.29** | **10.39** |
| XGBoost (baseline) | 13.45 | 10.39 |
| Transformer encoder | 35.72 | 28.43 |

**Key finding:** the Transformer significantly underperformed both other models — consistent with known literature on small time-series datasets, where Transformers' lack of built-in recurrence/locality biases leads to overfitting without large amounts of data (~17K training windows here). The CNN-LSTM, after proper regularization and validation-based early stopping, narrowly outperformed the XGBoost baseline while offering richer temporal representation learning.

## Pipeline

Raw sensor logs (train/test_FD001.txt) → Data cleaning & column parsing → RUL label engineering (capped at 125 cycles) → Normalization (MinMax scaling) → Sliding window construction (30-cycle windows) → Model training (CNN-LSTM / Transformer / XGBoost) → Evaluation (RMSE, MAE on held-out test engines) → Deployment (Streamlit web app)

### Model architecture — CNN-LSTM (best performer)
- 1D-CNN layer extracts local temporal patterns across sensor channels
- LSTM layer models longer-term degradation trends
- Dropout regularization + early stopping (validation-based) + LR scheduling
- Trained with gradient clipping for stability

### Model architecture — Transformer (tested, not selected)
- Lightweight encoder (2 layers, 4 attention heads) with sinusoidal positional encoding
- Mean-pooled across the full 30-cycle window (uses more temporal context than the CNN-LSTM's last-timestep approach)
- Included specifically to give an evidence-based answer to "why not use a Transformer" rather than an untested assumption

## MLOps Stack

- **Experiment tracking:** MLflow, hosted on [DagsHub](https://dagshub.com/mail2rahulghosh007-coder/rul_prediction_project) — every training run (hyperparameters, per-epoch loss curves, final metrics) logged and comparable across all three models
- **Data & model versioning:** DVC, with a full `dvc.yaml` pipeline — the entire pipeline runs end-to-end with a single `dvc repro`, with automatic caching of unchanged stages
- **Cloud storage:** raw data, processed data, and trained model artifacts stored via DVC remote on DagsHub

## Deployment

Two deployment paths are included, reflecting different real-world purposes:

**1. Live public demo** ([Streamlit Community Cloud](https://rul-prediction-project-curwtuldsgqrwxqdgapjrg.streamlit.app/)) — a lightweight, single-process app for quick access by anyone, no setup required. Supports two modes: browsing and predicting on real test engines from the dataset (with ground-truth comparison), or uploading your own 30-cycle sensor CSV for a live prediction.

**2. Containerized production-style setup** (Docker + FastAPI) — a REST API (`src/app.py`) serving the model behind a `/predict` endpoint, packaged in a `Dockerfile`, also published to Docker Hub for portability:

    docker pull dockerrahulma25m021/rul-prediction-app
    docker run -p 7860:7860 dockerrahulma25m021/rul-prediction-app

## Tech Stack

`Python` · `PyTorch` · `XGBoost` · `scikit-learn` · `FastAPI` · `Streamlit` · `Docker` · `MLflow` · `DVC` · `DagsHub`

## Repository Structure

    src/
      phase1_load_data.py            # Load raw sensor logs
      phase2_create_rul.py           # RUL label engineering
      phase3_preprocess_data.py      # Normalization
      phase4_sliding_window.py       # Sliding window construction
      phase5_model.py                # CNN-LSTM architecture
      phase6_train.py                # CNN-LSTM training (+ MLflow logging)
      phase7_evaluate.py             # CNN-LSTM evaluation
      phase8_transformer_model.py    # Transformer architecture
      phase9_train_transformer.py    # Transformer training (+ MLflow logging)
      phase10_evaluate_transformer.py
      phase11_baseline_xgboost.py    # XGBoost baseline (+ MLflow logging, full comparison)
      app.py                         # FastAPI inference service
      streamlit_app_standalone.py    # Deployed Streamlit app
    dvc.yaml                         # Full pipeline definition
    Dockerfile, requirements.txt     # Containerized deployment

## Running Locally

    # Full pipeline (data → training → evaluation, all 3 models)
    dvc repro

    # Or run the Streamlit app directly
    streamlit run src/streamlit_app_standalone.py

    # Or run the FastAPI backend
    cd src
    uvicorn app:app --reload

## Key Engineering Decisions

- **Fair model comparison:** all three models trained/evaluated on identical train/validation/test splits, with the same random seed, to ensure differences in performance reflect the architectures, not the data split
- **Regularization was necessary, not optional:** the initial CNN-LSTM (no validation split, fixed 10 epochs) was outperformed by XGBoost; after adding validation-based early stopping, dropout, and LR scheduling, RMSE improved ~11% and the model edged ahead of XGBoost
- **Metrics are saved to disk, not hardcoded:** each model's evaluation script writes its metrics to a JSON file, and the comparison script reads from those files, avoiding stale manually copy-pasted comparisons

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset (FD001 subset). [Source](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
