# 🤖 AI Analyst Agent

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

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
- 4GB+ RAM (8GB recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/govind104/agentic-rag-analyst.git
cd AgenticRAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

### Running Locally

```bash
# Terminal 1: Start FastAPI backend
python src/agent.py

# Terminal 2: Start Streamlit frontend
streamlit run src/app.py

# Terminal 3 (optional): Start MLflow
./mlflow_run.sh  # or: mlflow server --host 0.0.0.0 --port 5000
```

**Access:**
| Service | URL |
|---------|-----|
| 🖥️ Streamlit UI | http://localhost:8501 |
| 📖 FastAPI Docs | http://localhost:8001/docs |
| 📊 MLflow | http://localhost:5000 |
| 📈 Metrics | http://localhost:8001/metrics |

---

## 🐳 Docker Deployment

```bash
# Build and run all services
# Run from repository root
docker-compose -f docker/docker-compose.yml up --build

# Or run individually
docker build -t ai-analyst-agent -f docker/Dockerfile .
docker run -p 8501:8501 -p 8001:8001 ai-analyst-agent
```

---

## 📁 Project Structure

```
AgenticRAG/
├── src/
│   ├── retrieval/      # RAG tasks (Task1.py, Task2.py)
│   ├── agent.py        # FastAPI + LangGraph agent
│   ├── app.py          # Streamlit frontend
│   ├── data.py         # SQLite data layer
│   └── ethics.py       # Bias detection & guardrails
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   └── test.py         # Integration tests
├── docs/
│   └── README.md       # Documentation
├── .streamlit/         # Streamlit Cloud config
│   └── config.toml
├── requirements.txt    # Python dependencies
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

## 💬 Sample Queries

| Query | What It Does |
|-------|--------------|
| "Top 5 locations by fare" | Sum fares by location, show top 5 |
| "Bottom 10 locations by fare" | Sum fares by location, show bottom 10 |
| "Churn rate by region" | Average churn rate per region |
| "Average revenue by region" | Mean revenue grouped by region |
| "Trips by month" | Count trips per month |
| "Average fare by passengers" | Mean fare grouped by passenger count |

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

| Category | Technologies |
|----------|--------------|
| **GenAI/LLMs** | HuggingFace Transformers, Prompt Engineering |
| **RAG Systems** | Embedding Models, Vector Similarity, Top-K Retrieval |
| **Agents** | LangGraph State Machines, Tool Calling |
| **MLOps** | MLflow Tracking, Docker, Prometheus |
| **Backend** | FastAPI, Async Python, Queue/Batching |
| **Frontend** | Streamlit, Plotly, Responsive UI |
| **Data Engineering** | SQLite, Pandas, Synthetic Data Generation |
| **Ethical AI** | Bias Detection, Content Safety, Guardrails |

---

## 📊 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| p95 Latency | < 2s | ✅ ~200ms |
| Bias Threshold | < 0.05 | ✅ 0.0 (neutral queries) |
| Data Scale | 10k rows | ✅ 20k rows |
| Test Coverage | 100% | ✅ 8/8 suites |
| PRD Compliance | 100% | ✅ 98% |

---

## ✅ PRD Compliance Matrix

| Section | Requirement | Status | Implementation Details |
|---------|-------------|:------:|------------------------|
| **Core** | Agentic RAG (LangGraph) | ✅ | `src/agent.py` StateGraph with conditional edges |
| **Tools** | SQL / Retrieve / Viz / Bias | ✅ | `SQLQueryTool`, `RetrieveTool`, `VizTool`, `BiasTool` implemented |
| **Data** | NYC Taxi & Churn Data | ✅ | `src/data.py` synthetic generator (10k+ rows) |
| **Frontend** | Streamlit Chat + Dashboard | ✅ | `src/app.py` with sidebar conf & live metrics |
| **MLOps** | MLflow Integration | ✅ | Enabled in `src/agent.py` & `docker-compose.yml` |
| **Infra** | Docker / Prometheus | ✅ | Containerized + `/metrics` endpoint active |
| **Ethics** | Bias Detection | ✅ | `src/ethics.py` NLTK-based gender skew check |
| **Perf** | Latency < 2s | ✅ | Avg ~200ms (p95 < 500ms) on local test |
| **Compat** | CPU Fallback | ✅ | Uses `numpy` & `opt-125m` (no CUDA required) |

## 🙏 Acknowledgments

- University of Edinburgh - Machine Learning Systems Course
- HuggingFace - Transformers & Models
- Streamlit - Frontend Framework
- MLflow - Experiment Tracking
