# 🤖 AI Analyst Agent

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

> **Production-Ready RAG-Powered Data Copilot** with agentic capabilities, MLflow tracking, and ethical AI monitoring.

A full-stack GenAI demo showcasing modern ML/AI engineering practices for Data Science job applications.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Agentic RAG** | LangGraph state machine with SQL, retrieval, visualization tools |
| 📊 **SQL Analysis** | Natural language to SQL on NYC Taxi & Customer Churn data |
| 📈 **Auto Visualization** | Plotly charts generated from query results |
| ⚖️ **Ethical AI** | Bias detection, PII redaction, content guardrails |
| 📦 **MLflow Tracking** | Experiment logging with params, metrics, artifacts |
| 🔧 **Prometheus Metrics** | Live latency, throughput, bias score monitoring |
| 🐳 **Docker Ready** | One-command deployment with docker-compose |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone and enter directory
cd AgenticRAG

# Install dependencies
uv pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

### Running Locally

```bash
# Terminal 1: Start FastAPI backend
python agent.py

# Terminal 2: Start Streamlit frontend
streamlit run app.py

# Terminal 3 (optional): Start MLflow
./mlflow_run.sh
```

**Access:**
- Streamlit UI: http://localhost:8501
- FastAPI Docs: http://localhost:8001/docs
- MLflow: http://localhost:5000

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

This starts:
- **App** (Streamlit + FastAPI): ports 8501, 8001
- **MLflow**: port 5000

---

## 📁 Project Structure

```
AgenticRAG/
├── app.py              # Streamlit frontend (chat, dashboard, docs)
├── agent.py            # FastAPI + LangGraph agent
├── data.py             # SQLite data layer (NYC Taxi, Churn)
├── ethics.py           # Bias detection & guardrails
├── Task1.py            # GPU/CPU distance functions
├── Task2.py            # Original RAG FastAPI
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── docker-compose.yml  # Multi-service orchestration
└── mlflow_run.sh       # MLflow server script
```

---

## 🗄️ Data Schema

### NYC Taxi Trips (10,000 rows)
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Trip ID |
| pickup_date | TIMESTAMP | Pickup datetime |
| location | INTEGER | NYC taxi zone (1-265) |
| fare | FLOAT | Trip fare (USD) |
| passengers | INTEGER | Passenger count |

### Customer Churn (10,000 rows)
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Customer ID |
| region | TEXT | Geographic region |
| tenure | INTEGER | Months as customer |
| churn | INTEGER | Churned (1/0) |
| revenue | FLOAT | Revenue (USD) |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent` | POST | Main agent endpoint |
| `/rag` | POST | Legacy RAG (backward compatible) |
| `/metrics` | GET | Prometheus metrics |
| `/health` | GET | Health check |
| `/tables` | GET | Database schema info |

### Example Request

```bash
curl -X POST http://localhost:8001/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Top 5 locations by fare", "k": 10}'
```

---

## 🛠️ Skills Demonstrated

- **GenAI/LLMs**: HuggingFace Transformers, prompt engineering
- **RAG Systems**: Embedding models, vector similarity, retrieval
- **Agents**: LangGraph state machines, tool calling
- **MLOps**: MLflow tracking, Docker, Prometheus
- **Backend**: FastAPI, async Python, queue/batching
- **Frontend**: Streamlit, Plotly, responsive UI
- **Data Engineering**: SQLite, Pandas, data generation
- **Ethical AI**: Bias detection, content safety, guardrails

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| p95 Latency | < 2s |
| Bias Threshold | < 0.05 |
| Data Scale | 20k rows |
| Models | E5-large (embed), OPT-125m (LLM) |

---

## 📝 License

MIT License - See LICENSE file

---

## 👤 Author

**Govind Arun Nampoothiri**  
MSc Data Science, University of Edinburgh  
G.Arun-Nampoothiri@sms.ed.ac.uk
