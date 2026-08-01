# src/phase6_train.py
# UPDATED: added train/val split, early stopping, LR scheduler, gradient clipping
# UPDATED: added MLflow experiment tracking

import torch
import torch.nn as nn
import numpy as np
import mlflow
import dagshub
from phase5_model import CNN_LSTM_Model

# ---- Point MLflow to DagsHub instead of local storage ----
dagshub.init(repo_owner='mail2rahulghosh007-coder', repo_name='rul_prediction_project', mlflow=True)

# All models (CNN-LSTM, Transformer, XGBoost) log into this same experiment,
# so they can be compared side-by-side in the MLflow UI (now hosted on DagsHub)
mlflow.set_experiment("RUL_Prediction_CMAPSS_FD001")

# ---- Reproducibility ----
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load the sliding window data from Phase 4
X = np.load('data/X_train.npy')
y = np.load('data/y_train.npy')

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

print("Full dataset:", X_tensor.shape, y_tensor.shape)

# ---- Train / Validation split ----
# IMPORTANT: split by engine_id would be more rigorous (avoids windows from the
# same engine leaking between train/val), but for a straightforward first fix
# we do a random 85/15 split. Note this in your README as a known limitation.
num_samples = X_tensor.shape[0]
indices = torch.randperm(num_samples)
split = int(num_samples * 0.85)
train_idx, val_idx = indices[:split], indices[split:]

X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

print(f"Train samples: {X_train.shape[0]} | Val samples: {X_val.shape[0]}")

# ---- Model / Loss / Optimizer ----
num_features = X.shape[2]
window_size = X.shape[1]
model = CNN_LSTM_Model(num_features, window_size, dropout=0.3)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Reduce LR when validation loss plateaus -- helps LSTM converge better than a fixed LR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10  # early stopping patience
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# ---- Start MLflow run and log hyperparameters ----
mlflow.start_run(run_name="CNN_LSTM_training")
mlflow.set_tag("model_type", "CNN_LSTM")
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("max_epochs", MAX_EPOCHS)
mlflow.log_param("early_stopping_patience", PATIENCE)
mlflow.log_param("dropout", DROPOUT)
mlflow.log_param("learning_rate", LEARNING_RATE)
mlflow.log_param("weight_decay", WEIGHT_DECAY)
mlflow.log_param("window_size", window_size)
mlflow.log_param("num_features", num_features)
mlflow.log_param("train_samples", X_train.shape[0])
mlflow.log_param("val_samples", X_val.shape[0])

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

train_losses, val_losses = [], []

for epoch in range(MAX_EPOCHS):
    # ---- Training ----
    model.train()
    total_train_loss = 0
    permutation = torch.randperm(X_train.shape[0])

    for i in range(0, X_train.shape[0], BATCH_SIZE):
        idx = permutation[i:i+BATCH_SIZE]
        batch_X, batch_y = X_train[idx], y_train[idx]

        predictions = model(batch_X)
        loss = loss_function(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping -- prevents occasional exploding gradients in the LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_train_loss += loss.item() * batch_X.shape[0]

    avg_train_loss = total_train_loss / X_train.shape[0]

    # ---- Validation ----
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_function(val_predictions, y_val).item()

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # Log this epoch's metrics to MLflow (shows up as a live chart in the UI)
    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("learning_rate", current_lr, step=epoch)

    print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

    # ---- Early stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

# ---- Restore best model and save ----
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), 'models/cnn_lstm_model.pth')

np.save('data/train_losses.npy', np.array(train_losses))
np.save('data/val_losses.npy', np.array(val_losses))

print(f"\nBest validation loss: {best_val_loss:.4f}")
print("Model saved to models/cnn_lstm_model.pth")
print("Loss curves saved to data/train_losses.npy and data/val_losses.npy")

# ---- Log final results and model artifact to MLflow, then close the run ----
mlflow.log_metric("best_val_loss", best_val_loss)
mlflow.log_artifact('models/cnn_lstm_model.pth')
mlflow.end_run()
print("Logged run to MLflow (run: mlflow ui, then open http://localhost:5000)")