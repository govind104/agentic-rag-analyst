FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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

# Create startup script
RUN echo '#!/bin/bash\n\
# Start MLflow server in background\n\
python -m mlflow server --host 0.0.0.0 --port 5000 &\n\
\n\
# Start FastAPI backend in background\n\
python src/agent.py &\n\
\n\
# Wait for backend to be ready\n\
echo "Waiting for backend..."\n\
for i in {1..60}; do\n\
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then\n\
        echo "Backend ready!"\n\
        break\n\
    fi\n\
    sleep 2\n\
done\n\
\n\
# Start Streamlit (foreground)\n\
exec streamlit run src/app.py --server.port 7860 --server.address 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]
