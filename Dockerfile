FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Pre-download models at build time (critical for fast startup)
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    AutoTokenizer.from_pretrained('facebook/opt-125m'); \
    print('Models pre-downloaded successfully!')"

# Pre-build the database
RUN python -c "import sys; sys.path.insert(0, 'src'); from data import init_database; init_database(); print('Database pre-built successfully!')"

# Expose ports
EXPOSE 7860 8000 5000

# Create startup script as a separate file
COPY <<EOF /app/start.py
import subprocess
import time
import sys
import os

print("=" * 60)
print("AI ANALYST AGENT - DOCKER CONTAINER STARTING")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Files in /app: {os.listdir('/app')[:10]}")
print("=" * 60)

# Start MLflow server
print("[1/4] Starting MLflow server on port 5000...")
mlflow_proc = subprocess.Popen(
    [sys.executable, "-m", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
print("      MLflow process started.")

# Start FastAPI backend
print("[2/4] Starting FastAPI backend on port 8000...")
backend_proc = subprocess.Popen(
    [sys.executable, "src/agent.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)
print("      FastAPI process started.")

# Wait for backend health check
print("[3/4] Waiting for backend health check...")
import requests
ready = False
for i in range(120):
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code == 200:
            print(f"      Backend ready after {i+1} seconds!")
            ready = True
            break
    except:
        pass
    time.sleep(1)
    if i % 10 == 0 and i > 0:
        print(f"      Still waiting... ({i}s)")

if not ready:
    print("      WARNING: Backend did not become ready in 120s")
    if backend_proc.poll() is not None:
        out, _ = backend_proc.communicate()
        print(f"      Backend output: {out[:2000]}")

# Start Streamlit
print("[4/4] Starting Streamlit on port 7860...")
print("=" * 60)
subprocess.run([
    sys.executable, "-m", "streamlit", "run", "src/app.py",
    "--server.port", "7860",
    "--server.address", "0.0.0.0",
    "--server.headless", "true"
])
EOF

# Run the startup script
CMD ["python", "/app/start.py"]
