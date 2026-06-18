#!/usr/bin/env bash
set -e

APP_FILE="${APP_FILE:-app.py}"
PORT="${PORT:-8501}"
STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

echo "===== JENI STARTUP ====="
echo "Working directory: $(pwd)"
echo "Python: $(python --version)"
echo "App file: ${APP_FILE}"
echo "Port: ${PORT}"
echo "Address: ${STREAMLIT_SERVER_ADDRESS}"
echo "Directory listing:"
ls -la
echo "Checking Streamlit:"
python -m streamlit version
echo "Checking app file:"
test -f "${APP_FILE}"

mkdir -p /app/logging
mkdir -p /app/stores/sqlite/datamodels
mkdir -p /app/resources/images
mkdir -p /app/resources/audio

export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}"

echo "Launching Streamlit..."
exec python -m streamlit run "${APP_FILE}" \
  --server.address="${STREAMLIT_SERVER_ADDRESS}" \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false