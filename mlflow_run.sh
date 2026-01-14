#!/bin/bash
# ==============================================================================
# MLflow Tracking Server Startup Script
# Usage: ./mlflow_run.sh [--background]
# ==============================================================================

set -e

MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_BACKEND_STORE="${MLFLOW_BACKEND_STORE:-sqlite:///mlflow.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-./mlruns}"

echo "=========================================="
echo "  AI Analyst Agent - MLflow Server"
echo "=========================================="
echo "Host: $MLFLOW_HOST"
echo "Port: $MLFLOW_PORT"
echo "Backend Store: $MLFLOW_BACKEND_STORE"
echo "Artifact Root: $MLFLOW_ARTIFACT_ROOT"
echo "=========================================="

# Create artifact directory if it doesn't exist
mkdir -p "$MLFLOW_ARTIFACT_ROOT"

# Check if running in background mode
if [ "$1" == "--background" ]; then
    echo "Starting MLflow server in background..."
    nohup mlflow server \
        --host "$MLFLOW_HOST" \
        --port "$MLFLOW_PORT" \
        --backend-store-uri "$MLFLOW_BACKEND_STORE" \
        --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
        > mlflow.log 2>&1 &
    echo "MLflow server started with PID $!"
    echo "Logs: mlflow.log"
    echo "UI: http://localhost:$MLFLOW_PORT"
else
    echo "Starting MLflow server..."
    echo "UI: http://localhost:$MLFLOW_PORT"
    mlflow server \
        --host "$MLFLOW_HOST" \
        --port "$MLFLOW_PORT" \
        --backend-store-uri "$MLFLOW_BACKEND_STORE" \
        --default-artifact-root "$MLFLOW_ARTIFACT_ROOT"
fi
