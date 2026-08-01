FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by torch/pandas builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only build first (avoids downloading several GB of
# unnecessary NVIDIA GPU/CUDA libraries -- this container does CPU inference)
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies (better Docker layer caching --
# this layer only rebuilds if requirements.txt changes, not on every
# code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy trained model artifacts (scaler + CNN-LSTM weights)
COPY models/ ./models/

# Copy the raw test data (needed for the Streamlit "Test Engine" demo tab)
COPY data/test_FD001.txt ./data/test_FD001.txt
COPY data/RUL_FD001.txt ./data/RUL_FD001.txt

# Copy the startup script
COPY start.sh .
RUN chmod +x start.sh

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860
EXPOSE 7860

CMD ["./start.sh"]