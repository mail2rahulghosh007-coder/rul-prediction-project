# src/streamlit_app.py
# UI for the RUL Prediction project.
# Tab 1: pick a real test engine from the CMAPSS dataset (demo mode).
# Tab 2: upload/enter YOUR OWN engine's real 30-cycle sensor readings
#        (e.g. an engine you actually own and are monitoring) and get a
#        live RUL prediction from it.

import streamlit as st
import pandas as pd
import requests
import os
import io

API_URL = os.environ.get("RUL_API_URL", "http://127.0.0.1:8000/predict")
WINDOW_SIZE = 30

FEATURE_COLUMNS = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]

st.set_page_config(page_title="RUL Prediction — Turbofan Engine", page_icon="⚙️", layout="centered")

st.title("⚙️ Turbofan Engine RUL Predictor")
st.caption(
    "Predicts Remaining Useful Life (RUL) from the last 30 cycles of sensor "
    "readings, using a CNN-LSTM trained on NASA's CMAPSS FD001 dataset."
)

tab1, tab2 = st.tabs(["📊 Test Engine (from dataset)", "📤 Your Own Engine Data"])


def call_predict_api(cycles_payload):
    """Shared helper: sends a 30-cycle payload to the FastAPI backend."""
    payload = {"cycles": cycles_payload}
    with st.spinner("Calling the model..."):
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()


# ============================================================
# TAB 1 — Demo mode using real engines from the CMAPSS dataset
# ============================================================
with tab1:
    @st.cache_data
    def load_test_data():
        column_names = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
        column_names += [f'sensor_{i}' for i in range(1, 22)]

        test_df = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None)
        test_df = test_df.iloc[:, :26]
        test_df.columns = column_names

        true_rul = pd.read_csv('data/RUL_FD001.txt', header=None)
        true_rul.columns = ['RUL']
        return test_df, true_rul

    try:
        test_df, true_rul_df = load_test_data()
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False
        st.error(
            "Couldn't find `data/test_FD001.txt` or `data/RUL_FD001.txt`. "
            "Run this app from your project root: `streamlit run src/streamlit_app.py`"
        )

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

            cycles_payload = window_data[FEATURE_COLUMNS].to_dict(orient='records')

            try:
                result = call_predict_api(cycles_payload)
                predicted_rul = result['predicted_rul']

                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted RUL", f"{predicted_rul} cycles")
                col2.metric("True RUL", f"{true_rul} cycles")
                col3.metric("Error", f"{abs(predicted_rul - true_rul):.1f} cycles")

                if abs(predicted_rul - true_rul) < 15:
                    st.success("Prediction is close to the true RUL. ✅")
                else:
                    st.warning("Prediction differs notably from the true RUL.")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the prediction API. Make sure it's running:\n\n"
                    "```\ncd src\nuvicorn app:app --reload\n```"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")


# ============================================================
# TAB 2 — Bring your own engine's real 30-cycle sensor data
# ============================================================
with tab2:
    st.write(
        "If you have real sensor readings from an actual engine you're "
        "monitoring, upload the last **30 cycles** as a CSV here to get a "
        "live RUL prediction."
    )

    # Provide a ready-to-fill CSV template so the user knows the exact
    # column names/order the model expects
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
                st.error(f"Expected exactly {WINDOW_SIZE} rows, but found {len(user_df)}.")
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
                    cycles_payload = user_df[FEATURE_COLUMNS].to_dict(orient='records')
                    try:
                        result = call_predict_api(cycles_payload)
                        predicted_rul = result['predicted_rul']
                        st.metric("Predicted RUL", f"{predicted_rul} cycles")

                        if predicted_rul < 20:
                            st.error("⚠️ Low RUL — this engine may need maintenance soon.")
                        elif predicted_rul < 60:
                            st.warning("Moderate RUL — monitor this engine closely.")
                        else:
                            st.success("Healthy RUL range.")

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "Could not reach the prediction API. Make sure it's running:\n\n"
                            "```\ncd src\nuvicorn app:app --reload\n```"
                        )
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")

        except Exception as e:
            st.error(f"Couldn't read the CSV: {e}")

st.divider()
st.caption(
    "Model: CNN-LSTM (best performer vs. XGBoost baseline and Transformer variant). "
    "Full experiment tracking on MLflow/DagsHub, pipeline versioned with DVC."
)