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

# Create Python startup script (more portable than bash)
RUN echo 'import subprocess\n\
    import time\n\
    import sys\n\
    import requests\n\
    \n\
    print("Starting AI Analyst Agent...")\n\
    \n\
    # Start MLflow server\n\
    mlflow_proc = subprocess.Popen([sys.executable, "-m", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000"])\n\
    print("MLflow server starting...")\n\
    \n\
    # Start FastAPI backend\n\
    backend_proc = subprocess.Popen([sys.executable, "src/agent.py"])\n\
    print("FastAPI backend starting...")\n\
    \n\
    # Wait for backend to be ready\n\
    print("Waiting for backend...")\n\
    for i in range(120):\n\
    try:\n\
    resp = requests.get("http://localhost:8000/health", timeout=2)\n\
    if resp.status_code == 200:\n\
    print(f"Backend ready after {i+1} seconds!")\n\
    break\n\
    except:\n\
    pass\n\
    time.sleep(1)\n\
    if i % 10 == 0 and i > 0:\n\
    print(f"Still waiting... ({i}s)")\n\
    \n\
    # Start Streamlit (this blocks)\n\
    print("Starting Streamlit on port 7860...")\n\
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.headless", "true"])\n\
    ' > /app/start.py

# Run the startup script
CMD ["python", "/app/start.py"]
