# src/phase10_train_transformer.py
# Trains the Transformer RUL model using the exact same discipline as the
# fixed CNN-LSTM training (phase6_train.py): validation split, early
# stopping, LR scheduling, gradient clipping -- so the comparison is fair.

import torch
import torch.nn as nn
import numpy as np
import json
import mlflow
import dagshub
from phase8_transformer_model import TransformerRULModel

dagshub.init(repo_owner='mail2rahulghosh007-coder', repo_name='rul_prediction_project', mlflow=True)
mlflow.set_experiment("RUL_Prediction_CMAPSS_FD001")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

X = np.load('data/X_train.npy')
y = np.load('data/y_train.npy')

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

print("Full dataset:", X_tensor.shape, y_tensor.shape)

# Same 85/15 split logic and same SEED as phase6_train.py, so both models
# are trained/validated on the exact same split -- apples-to-apples.
num_samples = X_tensor.shape[0]
indices = torch.randperm(num_samples)
split = int(num_samples * 0.85)
train_idx, val_idx = indices[:split], indices[split:]

X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

print(f"Train samples: {X_train.shape[0]} | Val samples: {X_val.shape[0]}")

num_features = X.shape[2]
window_size = X.shape[1]
model = TransformerRULModel(num_features, window_size, dropout=0.3)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
DROPOUT = 0.3
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2

mlflow.start_run(run_name="Transformer_training")
mlflow.set_tag("model_type", "Transformer")
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("max_epochs", MAX_EPOCHS)
mlflow.log_param("early_stopping_patience", PATIENCE)
mlflow.log_param("dropout", DROPOUT)
mlflow.log_param("d_model", D_MODEL)
mlflow.log_param("nhead", NHEAD)
mlflow.log_param("num_layers", NUM_LAYERS)
mlflow.log_param("window_size", window_size)
mlflow.log_param("num_features", num_features)
mlflow.log_param("train_samples", X_train.shape[0])
mlflow.log_param("val_samples", X_val.shape[0])

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None
train_losses, val_losses = [], []

for epoch in range(MAX_EPOCHS):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_train_loss += loss.item() * batch_X.shape[0]

    avg_train_loss = total_train_loss / X_train.shape[0]

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = loss_function(val_predictions, y_val).item()

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("learning_rate", current_lr, step=epoch)

    print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

model.load_state_dict(best_model_state)
torch.save(model.state_dict(), 'models/transformer_model.pth')

np.save('data/transformer_train_losses.npy', np.array(train_losses))
np.save('data/transformer_val_losses.npy', np.array(val_losses))

print(f"\nBest validation loss: {best_val_loss:.4f}")
print("Model saved to models/transformer_model.pth")

mlflow.log_metric("best_val_loss", best_val_loss)
mlflow.log_artifact('models/transformer_model.pth')
mlflow.end_run()
print("Logged run to MLflow")