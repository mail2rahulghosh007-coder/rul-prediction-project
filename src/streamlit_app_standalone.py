# src/streamlit_app_standalone.py
# Deployment version: loads the model DIRECTLY inside Streamlit instead of
# calling a separate FastAPI process. This keeps the deployed container to
# a single lightweight process, important for free-tier RAM limits (HF
# Spaces free CPU tier, Render free tier, etc.) where running two full
# Python services (uvicorn + streamlit) at once risks running out of memory.
#
# The FastAPI app (src/app.py) is still kept in the repo as a standalone
# API for local use / demonstration -- this file just doesn't depend on it.

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import io

from phase5_model import CNN_LSTM_Model

WINDOW_SIZE = 30

FEATURE_COLUMNS = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]
NUM_FEATURES = len(FEATURE_COLUMNS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'cnn_lstm_model.pth')

st.set_page_config(page_title="RUL Prediction — Turbofan Engine", page_icon="⚙️", layout="centered")


@st.cache_resource
def load_model_and_scaler():
    """Loaded once and cached across reruns/users -- avoids reloading the
    model on every interaction, which matters a lot on limited free-tier RAM/CPU."""
    scaler = joblib.load(SCALER_PATH)
    model = CNN_LSTM_Model(num_features=NUM_FEATURES, window_size=WINDOW_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, scaler


def predict_rul(cycles_df, model, scaler):
    raw_matrix = cycles_df[FEATURE_COLUMNS].values
    scaled_matrix = scaler.transform(raw_matrix)
    input_tensor = torch.tensor(scaled_matrix, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return round(max(0.0, min(prediction, 125.0)), 1)


try:
    model, scaler = load_model_and_scaler()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model/scaler: {e}")

st.title("⚙️ Turbofan Engine RUL Predictor")
st.caption(
    "Predicts Remaining Useful Life (RUL) from the last 30 cycles of sensor "
    "readings, using a CNN-LSTM trained on NASA's CMAPSS FD001 dataset."
)

if model_loaded:
    tab1, tab2 = st.tabs(["📊 Test Engine (from dataset)", "📤 Your Own Engine Data"])

    with tab1:
        @st.cache_data
        def load_test_data():
            column_names = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
            column_names += [f'sensor_{i}' for i in range(1, 22)]
            test_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'test_FD001.txt'),
                                   sep='\s+', header=None)
            test_df = test_df.iloc[:, :26]
            test_df.columns = column_names
            true_rul = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'RUL_FD001.txt'), header=None)
            true_rul.columns = ['RUL']
            return test_df, true_rul

        try:
            test_df, true_rul_df = load_test_data()
            data_loaded = True
        except FileNotFoundError:
            data_loaded = False
            st.error("Test data files not found in this deployment.")

        if data_loaded:
            engine_ids = sorted(test_df['engine_id'].unique())
            selected_engine = st.selectbox("Select a test engine", engine_ids, index=0)

            engine_data = test_df[test_df['engine_id'] == selected_engine].reset_index(drop=True)
            num_cycles = len(engine_data)
            true_rul = min(true_rul_df.iloc[selected_engine - 1]['RUL'], 125)

            st.write(f"This engine has **{num_cycles}** recorded cycles in the test set.")

            st.subheader("Sensor trend (last 30 cycles used for prediction)")
            display_window = engine_data.tail(WINDOW_SIZE)[['cycle', 'sensor_2', 'sensor_11', 'sensor_15']]
            st.line_chart(display_window.set_index('cycle'))

            if st.button("🔮 Predict RUL", type="primary", key="predict_dataset"):
                if num_cycles < WINDOW_SIZE:
                    padding_needed = WINDOW_SIZE - num_cycles
                    first_row = engine_data.iloc[0:1]
                    padding = pd.concat([first_row] * padding_needed, ignore_index=True)
                    window_data = pd.concat([padding, engine_data], ignore_index=True)
                else:
                    window_data = engine_data.tail(WINDOW_SIZE)

                predicted_rul = predict_rul(window_data, model, scaler)

                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted RUL", f"{predicted_rul} cycles")
                col2.metric("True RUL", f"{true_rul} cycles")
                col3.metric("Error", f"{abs(predicted_rul - true_rul):.1f} cycles")

                if abs(predicted_rul - true_rul) < 15:
                    st.success("Prediction is close to the true RUL. ✅")
                else:
                    st.warning("Prediction differs notably from the true RUL.")

    with tab2:
        st.write(
            "If you have real sensor readings from an actual engine you're "
            "monitoring, upload the last **30 cycles** as a CSV here."
        )

        template_df = pd.DataFrame(columns=['cycle'] + FEATURE_COLUMNS)
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download CSV template (fill in 30 rows)",
            data=csv_buffer.getvalue(),
            file_name="engine_data_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader(
            "Upload your filled-in CSV (must have exactly 30 rows, oldest cycle first)",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                missing_cols = [c for c in FEATURE_COLUMNS if c not in user_df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                elif len(user_df) != WINDOW_SIZE:
                    st.error(f"Expected exactly {WINDOW_SIZE} rows, found {len(user_df)}.")
                else:
                    st.success(f"Loaded {len(user_df)} cycles. Preview:")
                    st.dataframe(user_df[FEATURE_COLUMNS].head())

                    st.subheader("Your sensor trend")
                    if 'cycle' in user_df.columns:
                        chart_df = user_df[['cycle', 'sensor_2', 'sensor_11', 'sensor_15']].set_index('cycle')
                    else:
                        chart_df = user_df[['sensor_2', 'sensor_11', 'sensor_15']]
                    st.line_chart(chart_df)

                    if st.button("🔮 Predict RUL for this engine", type="primary", key="predict_upload"):
                        predicted_rul = predict_rul(user_df, model, scaler)
                        st.metric("Predicted RUL", f"{predicted_rul} cycles")

                        if predicted_rul < 20:
                            st.error("⚠️ Low RUL — this engine may need maintenance soon.")
                        elif predicted_rul < 60:
                            st.warning("Moderate RUL — monitor this engine closely.")
                        else:
                            st.success("Healthy RUL range.")
            except Exception as e:
                st.error(f"Couldn't read the CSV: {e}")

st.divider()
st.caption(
    "Model: CNN-LSTM (best performer vs. XGBoost baseline and Transformer variant). "
    "Full experiment tracking on MLflow/DagsHub, pipeline versioned with DVC. "
    "A separate FastAPI backend (src/app.py) is also included in this repo for API-based serving."
)