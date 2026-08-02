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
- **Data & model versioning:** DVC, with a full `dvc.yaml` pipeline — the entire pipeline (data loading → preprocessing → training → evaluation, all 3 models) runs end-to-end with a single `dvc repro`, with automatic caching of unchanged stages
- **Cloud storage:** raw data, processed data, and trained model artifacts stored via DVC remote on DagsHub

## Deployment

Two deployment paths are included, reflecting different real-world purposes:

1. **Live public demo** ([Streamlit Community Cloud](https://rul-prediction-project-curwtuldsgqrwxqdgapjrg.streamlit.app/)) — a lightweight, single-process app for quick access by anyone, no setup required. Supports two modes:
   - Browse and predict on real test engines from the dataset (with ground-truth comparison)
   - Upload your own 30-cycle sensor CSV for a live prediction
2. **Containerized production-style setup** (Docker + FastAPI) — a REST API (`src/app.py`) serving the model behind a `/predict` endpoint, packaged in a `Dockerfile`, also published to Docker Hub for portability:
   docker pull dockerrahulma25m021/rul-prediction-app
docker run -p 7860:7860 dockerrahulma25m021/rul-prediction-app
