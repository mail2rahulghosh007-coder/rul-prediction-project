#!/bin/bash
# Starts both services in one container:
#  - FastAPI backend on port 8000 (internal only, called by Streamlit)
#  - Streamlit frontend on port 7860 (the port exposed to the outside world,
#    matching Hugging Face Spaces' default Docker port convention)

set -e

echo "Starting FastAPI backend..."
cd /app/src
uvicorn app:app --host 127.0.0.1 --port 8000 &

# Give the API a moment to start before Streamlit tries to call it
sleep 5

echo "Starting Streamlit frontend..."
cd /app
streamlit run src/streamlit_app.py --server.port 7860 --server.address 0.0.0.0