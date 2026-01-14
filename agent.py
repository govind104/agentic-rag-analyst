"""
AI Analyst Agent - Enhanced Backend
FastAPI + LangGraph agent with tools for SQL, RAG retrieval, visualization, and bias checking.
"""

# Suppress known warnings before any imports
import warnings
import os
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json
import time
import queue
from threading import Thread
from contextlib import asynccontextmanager
from typing import TypedDict, Annotated, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# LangChain & LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Transformers
from transformers import AutoTokenizer, AutoModel, pipeline

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# MLflow tracking
import mlflow
from mlflow.tracking import MlflowClient

# Local imports
from data import run_sql, get_table_info, init_database

# ==============================================================================
# Configuration
# ==============================================================================
MAX_BATCH_SIZE = 8
MAX_WAITING_TIME = 0.1
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
LLM_MODEL_NAME = "facebook/opt-125m"  # Lightweight for CPU

# ==============================================================================
# Prometheus Metrics
# ==============================================================================
REQUEST_COUNT = Counter("agent_requests_total", "Total agent requests")
REQUEST_LATENCY = Histogram("agent_request_latency_seconds", "Request latency", buckets=[0.1, 0.5, 1, 2, 5, 10])
QUEUE_SIZE = Gauge("agent_queue_size", "Current queue size")
BIAS_SCORE = Gauge("agent_bias_score", "Latest bias score")
BATCH_SIZE = Histogram("agent_batch_size", "Batch sizes processed", buckets=[1, 2, 4, 8, 16])

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "ai-analyst-agent"

# ==============================================================================
# Agent State Schema
# ==============================================================================
class AgentState(TypedDict):
    """LangGraph state for the AI Analyst Agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    sql_query: Optional[str]
    sql_result: Optional[str]
    retrieved_docs: list[str]
    viz_code: Optional[str]
    narrative: Optional[str]
    bias_score: float
    session_id: str
    error: Optional[str]


# ==============================================================================
# Models & Embeddings (Lazy Loading)
# ==============================================================================
class ModelManager:
    """Manages model loading and inference."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        if self._initialized:
            return
        
        print("Loading embedding model...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        self.embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        
        print("Loading LLM...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, padding_side="left")
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        device = 0 if torch.cuda.is_available() else -1
        self.llm_pipeline = pipeline(
            "text-generation",
            model=LLM_MODEL_NAME,
            tokenizer=self.llm_tokenizer,
            device=device,
            max_length=200,
            max_new_tokens=80,
            truncation=True,
            do_sample=False,
            repetition_penalty=2.0,
            batch_size=MAX_BATCH_SIZE
        )
        
        # Document store for RAG
        self.documents = self._load_documents()
        self.doc_embeddings = np.vstack([self.get_embedding(doc) for doc in self.documents])
        
        self._initialized = True
        print("Models loaded successfully!")
    
    def _load_documents(self) -> list[str]:
        """Load documents for RAG retrieval."""
        # Base documents
        docs = [
            "NYC Taxi data contains trip records with pickup dates, locations, fares, and passenger counts.",
            "Customer churn data tracks customer retention with region, tenure, churn status, and revenue.",
            "The trips table has columns: id, pickup_date, location, fare, passengers.",
            "The customers table has columns: id, region, tenure, churn, revenue.",
            "High tenure customers (>36 months) typically have lower churn rates around 15%.",
            "The average taxi fare follows an exponential distribution with a minimum of $2.50.",
            "NYC has 265 taxi zones identified by location IDs.",
            "Customer churn rate across all regions is approximately 27%.",
            "To analyze fare by location, use: SELECT location, AVG(fare) FROM trips GROUP BY location",
            "To analyze churn by region, use: SELECT region, AVG(churn)*100 as churn_rate FROM customers GROUP BY region"
        ]
        return docs
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        inputs = self.embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    def generate(self, prompt: str) -> str:
        """Generate text using the LLM."""
        result = self.llm_pipeline(prompt)
        return result[0]["generated_text"]


models = ModelManager()


# ==============================================================================
# Agent Tools (CPU-based using NumPy)
# ==============================================================================
def cpu_top_k_retrieval(query_emb: np.ndarray, doc_embs: np.ndarray, k: int = 5, metric: str = "l2") -> list[int]:
    """CPU-based top-K retrieval using NumPy (from Task1.py cpu_ functions)."""
    query_emb = np.asarray(query_emb, dtype=np.float32)
    doc_embs = np.asarray(doc_embs, dtype=np.float32)
    
    if query_emb.ndim == 1:
        query_emb = query_emb[None, :]
    
    if metric == "l2":
        distances = np.sum((query_emb - doc_embs) ** 2, axis=1)
    elif metric == "cosine":
        dot = np.sum(query_emb * doc_embs, axis=1)
        norm_q = np.linalg.norm(query_emb)
        norm_d = np.linalg.norm(doc_embs, axis=1)
        distances = 1 - (dot / (norm_q * norm_d + 1e-8))
    elif metric == "dot":
        distances = -np.sum(query_emb * doc_embs, axis=1)
    elif metric == "manhattan":
        distances = np.sum(np.abs(query_emb - doc_embs), axis=1)
    else:
        distances = np.sum((query_emb - doc_embs) ** 2, axis=1)
    
    indices = np.argsort(distances)[:k]
    return indices.tolist()


class SQLQueryTool:
    """Tool for executing SQL queries."""
    
    @staticmethod
    def run(query: str) -> dict:
        """Execute SQL query and return results."""
        try:
            # Basic SQL injection prevention
            forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
            query_upper = query.upper()
            for f in forbidden:
                if f in query_upper:
                    return {"error": f"Forbidden SQL operation: {f}", "data": None}
            
            df = run_sql(query)
            return {
                "data": df.to_dict(orient="records"),
                "columns": list(df.columns),
                "row_count": len(df),
                "error": None
            }
        except Exception as e:
            return {"error": str(e), "data": None}


class RetrieveTool:
    """Tool for RAG document retrieval."""
    
    @staticmethod
    def run(query: str, k: int = 5, metric: str = "l2") -> list[str]:
        """Retrieve top-K relevant documents."""
        query_emb = models.get_embedding(query)
        indices = cpu_top_k_retrieval(query_emb, models.doc_embeddings, k=k, metric=metric)
        return [models.documents[i] for i in indices]


class VizTool:
    """Tool for generating Plotly visualization code."""
    
    @staticmethod
    def run(data: list[dict], chart_type: str = "bar", x_col: str = None, y_col: str = None) -> dict:
        """Generate Plotly visualization specification."""
        if not data:
            return {"error": "No data provided", "plotly_json": None}
        
        df = pd.DataFrame(data)
        columns = list(df.columns)
        
        # Auto-detect columns if not specified
        if x_col is None:
            x_col = columns[0]
        if y_col is None:
            y_col = columns[1] if len(columns) > 1 else columns[0]
        
        # Convert x column to string for proper labeling
        df[x_col] = df[x_col].astype(str)
        
        # Get values
        x_values = df[x_col].tolist()
        y_values = df[y_col].tolist() if y_col != x_col else df[x_col].tolist()
        
        # Create readable x-axis labels
        if x_col.lower() in ['location', 'id', 'zone']:
            x_labels = [f"Loc {v}" for v in x_values]
            tick_text = x_labels
        elif x_col.lower() == 'region':
            x_labels = [str(v).title() for v in x_values]
            tick_text = x_labels
        else:
            x_labels = x_values
            tick_text = [str(v) for v in x_values]
        
        # Generate Plotly JSON spec with improved labels
        plotly_spec = {
            "data": [{
                "type": chart_type,
                "x": list(range(len(x_labels))),
                "y": y_values,
                "name": y_col,
                "text": [f"{v:.2f}" if isinstance(v, float) else str(v) for v in y_values],
                "textposition": "auto",
                "hovertext": [f"{tick_text[i]}: {y_values[i]:.2f}" if isinstance(y_values[i], float) else f"{tick_text[i]}: {y_values[i]}" for i in range(len(y_values))]
            }],
            "layout": {
                "title": f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                "xaxis": {
                    "title": x_col.replace('_', ' ').title(),
                    "tickmode": "array",
                    "tickvals": list(range(len(tick_text))),
                    "ticktext": tick_text,
                    "tickangle": 0  # Horizontal labels
                },
                "yaxis": {"title": y_col.replace('_', ' ').title()},
                "template": "plotly_dark",
                "showlegend": False
            }
        }
        
        return {"plotly_json": plotly_spec, "error": None}


class BiasTool:
    """Tool for checking bias in generated text."""
    
    # Gender-associated word lists (simplified)
    MALE_WORDS = {"he", "him", "his", "man", "men", "male", "boy", "father", "son", "brother", "husband"}
    FEMALE_WORDS = {"she", "her", "hers", "woman", "women", "female", "girl", "mother", "daughter", "sister", "wife"}
    
    @staticmethod
    def run(text: str) -> dict:
        """Compute demographic bias score for text."""
        if not text:
            return {"bias_score": 0.0, "details": "No text provided"}
        
        words = set(text.lower().split())
        
        male_count = len(words & BiasTool.MALE_WORDS)
        female_count = len(words & BiasTool.FEMALE_WORDS)
        total_gendered = male_count + female_count
        
        if total_gendered == 0:
            bias_score = 0.0
            details = "No gendered language detected"
        else:
            # Bias = deviation from 50/50 balance
            bias_score = abs(male_count - female_count) / total_gendered
            details = f"Male terms: {male_count}, Female terms: {female_count}"
        
        return {"bias_score": round(bias_score, 4), "details": details}


# ==============================================================================
# LangGraph Agent Definition
# ==============================================================================
def create_agent_graph():
    """Create the LangGraph state machine for the AI Analyst Agent."""
    
    def classify_query(state: AgentState) -> AgentState:
        """Classify query and decide routing."""
        query = state["query"].lower()
        
        # Check if query needs SQL
        sql_keywords = ["top", "average", "sum", "count", "group", "by", "where", "select", 
                       "fare", "revenue", "churn", "location", "region", "trips", "customers"]
        needs_sql = any(kw in query for kw in sql_keywords)
        
        if needs_sql:
            state["sql_query"] = "PENDING"
        else:
            state["sql_query"] = None
        
        return state
    
    def generate_sql(state: AgentState) -> AgentState:
        """Generate and execute SQL query based on natural language."""
        if state.get("sql_query") != "PENDING":
            return state
        
        query = state["query"]
        query_lower = query.lower()
        
        # Determine table and columns
        if any(kw in query_lower for kw in ["churn", "customer", "region", "tenure", "revenue"]):
            table = "customers"
            default_group = "region"
            default_agg = "revenue"
        else:
            table = "trips"
            default_group = "location"
            default_agg = "fare"
        
        # Determine aggregation function
        if "average" in query_lower or "avg" in query_lower:
            agg_func = "AVG"
        elif "count" in query_lower or "number" in query_lower:
            agg_func = "COUNT"
        elif "total" in query_lower or "sum" in query_lower:
            agg_func = "SUM"
        else:
            agg_func = "SUM"  # Default for "top by fare" type queries
        
        # Determine order direction
        if "bottom" in query_lower or "lowest" in query_lower or "least" in query_lower or "worst" in query_lower:
            order = "ASC"
        else:
            order = "DESC"  # Default to top/highest
        
        # Determine limit
        limit = 5  # Default
        for word in query_lower.split():
            if word.isdigit():
                limit = int(word)
                break
        
        # Determine what to group by
        if "passenger" in query_lower:
            group_col = "passengers"
        elif "tenure" in query_lower:
            group_col = "tenure"
        elif "region" in query_lower:
            group_col = "region"
        elif "location" in query_lower or "zone" in query_lower:
            group_col = "location"
        elif "month" in query_lower or "time" in query_lower or "date" in query_lower:
            if table == "trips":
                group_col = "strftime('%Y-%m', pickup_date)"
            else:
                group_col = default_group
        else:
            group_col = default_group
        
        # Determine what to aggregate
        if "fare" in query_lower:
            agg_col = "fare"
        elif "revenue" in query_lower:
            agg_col = "revenue"
        elif "churn" in query_lower:
            agg_col = "churn"
            agg_func = "AVG"  # Churn rate is always average
        elif "trip" in query_lower or "count" in query_lower:
            agg_col = "*"
            agg_func = "COUNT"
        else:
            agg_col = default_agg
        
        # Build the SQL query
        if "churn" in query_lower and "rate" in query_lower:
            sql = f"SELECT {group_col}, AVG(churn) * 100 as churn_rate, COUNT(*) as total FROM {table} GROUP BY {group_col} ORDER BY churn_rate {order} LIMIT {limit}"
        elif agg_col == "*":
            sql = f"SELECT {group_col}, COUNT(*) as count FROM {table} GROUP BY {group_col} ORDER BY count {order} LIMIT {limit}"
        else:
            agg_name = f"{agg_func.lower()}_{agg_col}"
            sql = f"SELECT {group_col}, {agg_func}({agg_col}) as {agg_name} FROM {table} GROUP BY {group_col} ORDER BY {agg_name} {order} LIMIT {limit}"
        
        result = SQLQueryTool.run(sql)
        state["sql_query"] = sql
        state["sql_result"] = json.dumps(result["data"][:20]) if result["data"] else None
        
        if result["error"]:
            state["error"] = result["error"]
        
        return state
    
    def retrieve_docs(state: AgentState) -> AgentState:
        """Retrieve relevant documents for context."""
        query = state["query"]
        docs = RetrieveTool.run(query, k=5, metric="l2")
        state["retrieved_docs"] = docs
        return state
    
    def generate_viz(state: AgentState) -> AgentState:
        """Generate visualization if we have SQL results."""
        if state.get("sql_result"):
            try:
                data = json.loads(state["sql_result"])
                if data and len(data) > 0:
                    viz_result = VizTool.run(data)
                    state["viz_code"] = json.dumps(viz_result.get("plotly_json"))
            except:
                state["viz_code"] = None
        return state
    
    def generate_narrative(state: AgentState) -> AgentState:
        """Generate deterministic narrative from data (no LLM)."""
        query = state["query"]
        narrative = f"Query processed: {query}. See visualization for details."
        
        if state.get("sql_result"):
            try:
                data = json.loads(state["sql_result"])
                if data and len(data) > 0:
                    num_results = len(data)
                    first_row = data[0]
                    keys = list(first_row.keys())
                    
                    if len(keys) >= 2:
                        # Identify grouping and value columns dynamically
                        group_col = keys[0]  # Usually first col is grouping (e.g., location, region)
                        val_col = keys[1]    # Usually second is metric (e.g., total_fare, avg_revenue)
                        
                        group_val = first_row[group_col]
                        val_val = first_row[val_col]
                        
                        # Format value
                        if isinstance(val_val, (int, float)):
                            val_str = f"{val_val:.2f}"
                        else:
                            val_str = str(val_val)
                        
                        # Select template based on column names
                        if "revenue" in val_col.lower():
                            template = f"Analysis: Found {num_results} results. Top result: {group_col} '{group_val}' generated highest revenue of ${val_str}. This segment leads in financial performance."
                        elif "churn" in val_col.lower():
                            template = f"Analysis: Found {num_results} results. Top result: {group_col} '{group_val}' has the highest churn rate of {val_str}. This segment requires retention focus."
                        elif "fare" in val_col.lower():
                            template = f"Analysis: Found {num_results} results. Top result: {group_col} '{group_val}' recorded the highest fare total of ${val_str}. High demand segment identified."
                        elif "count" in val_col.lower() or "total" in val_col.lower():
                            template = f"Analysis: Found {num_results} results. Top result: {group_col} '{group_val}' has {val_str} total records."
                        else:
                            # Generic fallback for other metrics
                            template = f"Analysis: Found {num_results} results. Top result: {group_col} '{group_val}' with {val_col} {val_str}. See chart for full breakdown."
                            
                        narrative = template
                    else:
                        narrative = f"Analysis: Found {num_results} results. Top result: {first_row}. See chart for details."
                else:
                    narrative = "Analysis: No data found matching your criteria."
            except Exception as e:
                narrative = f"Analysis completed. (Error parsing details: {str(e)})"
        
        state["narrative"] = narrative
        return state
    
    def check_bias(state: AgentState) -> AgentState:
        """Check for bias in generated narrative."""
        narrative = state.get("narrative", "")
        bias_result = BiasTool.run(narrative)
        state["bias_score"] = bias_result["bias_score"]
        BIAS_SCORE.set(bias_result["bias_score"])
        return state
    
    def should_generate_sql(state: AgentState) -> str:
        """Conditional edge: check if SQL is needed."""
        if state.get("sql_query") == "PENDING":
            return "generate_sql"
        return "retrieve_docs"
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify", classify_query)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("generate_viz", generate_viz)
    workflow.add_node("generate_narrative", generate_narrative)
    workflow.add_node("check_bias", check_bias)
    
    # Add edges
    workflow.set_entry_point("classify")
    workflow.add_conditional_edges("classify", should_generate_sql, {
        "generate_sql": "generate_sql",
        "retrieve_docs": "retrieve_docs"
    })
    workflow.add_edge("generate_sql", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "generate_viz")
    workflow.add_edge("generate_viz", "generate_narrative")
    workflow.add_edge("generate_narrative", "check_bias")
    workflow.add_edge("check_bias", END)
    
    return workflow.compile()


# ==============================================================================
# FastAPI Application
# ==============================================================================
class AgentRequest(BaseModel):
    """Request model for the agent endpoint."""
    query: str = Field(..., description="User query")
    k: int = Field(default=10, ge=1, le=50, description="Number of documents to retrieve")
    metric: str = Field(default="l2", description="Distance metric (l2, cosine, dot, manhattan)")
    session_id: Optional[str] = Field(default=None, description="Session ID for tracking")


class AgentResponse(BaseModel):
    """Response model for the agent endpoint."""
    query: str
    sql_query: Optional[str]
    sql_result: Optional[Any]
    narrative: str
    viz_code: Optional[str]
    bias_score: float
    latency_ms: float
    session_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("Initializing AI Analyst Agent...")
    
    # Initialize database
    init_database()
    
    # Initialize models
    models.initialize()
    
    # Create agent graph
    app.state.agent = create_agent_graph()
    app.state.request_queue = queue.Queue()
    
    # Setup MLflow (non-blocking - disabled if MLflow not running)
    app.state.mlflow_enabled = False
    try:
        import socket
        # Quick connection check to MLflow
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # 1 second timeout
        result = sock.connect_ex(('localhost', 5000))
        sock.close()
        
        if result == 0:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            app.state.mlflow_enabled = True
            print(f"MLflow tracking enabled: {MLFLOW_TRACKING_URI}")
        else:
            print("MLflow server not running - tracking disabled")
    except Exception as e:
        print(f"MLflow tracking disabled: {e}")
    
    print("AI Analyst Agent ready!")
    yield
    
    print("Shutting down AI Analyst Agent...")


app = FastAPI(
    title="AI Analyst Agent",
    description="RAG-Powered Data Copilot with LangGraph",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    """
    Main agent endpoint for processing analytical queries.
    
    Supports:
    - SQL queries on NYC Taxi and Customer Churn data
    - RAG-based document retrieval
    - Automated visualization generation
    - Bias checking on outputs
    """
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    try:
        # Initialize state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=request.query)],
            "query": request.query,
            "sql_query": None,
            "sql_result": None,
            "retrieved_docs": [],
            "viz_code": None,
            "narrative": None,
            "bias_score": 0.0,
            "session_id": request.session_id or f"session_{int(time.time())}",
            "error": None
        }
        
        # Run agent
        QUEUE_SIZE.set(app.state.request_queue.qsize())
        final_state = app.state.agent.invoke(initial_state)
        
        latency = (time.time() - start_time) * 1000
        REQUEST_LATENCY.observe(latency / 1000)
        
        # Parse SQL result for response
        sql_result = None
        if final_state.get("sql_result"):
            try:
                sql_result = json.loads(final_state["sql_result"])
            except:
                sql_result = final_state["sql_result"]
        
        # Log to MLflow
        if getattr(app.state, 'mlflow_enabled', False):
            try:
                with mlflow.start_run(run_name=f"query_{final_state['session_id']}"):
                    # Log parameters
                    mlflow.log_param("query", request.query[:100])
                    mlflow.log_param("k", request.k)
                    mlflow.log_param("metric", request.metric)
                    mlflow.log_param("session_id", final_state["session_id"])
                    
                    # Log metrics
                    mlflow.log_metric("latency_ms", latency)
                    mlflow.log_metric("bias_score", final_state.get("bias_score", 0.0))
                    mlflow.log_metric("docs_retrieved", len(final_state.get("retrieved_docs", [])))
                    
                    # Log trace artifact
                    trace_data = {
                        "query": request.query,
                        "sql_query": final_state.get("sql_query"),
                        "narrative": final_state.get("narrative"),
                        "bias_score": final_state.get("bias_score"),
                        "latency_ms": latency,
                        "timestamp": datetime.now().isoformat()
                    }
                    mlflow.log_dict(trace_data, "trace.json")
            except Exception as mlflow_err:
                print(f"MLflow logging error: {mlflow_err}")
        
        return AgentResponse(
            query=request.query,
            sql_query=final_state.get("sql_query"),
            sql_result=sql_result,
            narrative=final_state.get("narrative", "Unable to generate response."),
            viz_code=final_state.get("viz_code"),
            bias_score=final_state.get("bias_score", 0.0),
            latency_ms=round(latency, 2),
            session_id=final_state["session_id"]
        )
        
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        REQUEST_LATENCY.observe(latency / 1000)
        
        # Log error to MLflow
        if getattr(app.state, 'mlflow_enabled', False):
            try:
                with mlflow.start_run(run_name=f"error_{int(time.time())}"):
                    mlflow.log_param("query", request.query[:100])
                    mlflow.log_param("error", str(e)[:200])
                    mlflow.log_metric("latency_ms", latency)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/tables")
async def get_tables():
    """Get information about available database tables."""
    return get_table_info()


# Keep original /rag endpoint for backward compatibility
class QueryRequest(BaseModel):
    query: str
    k: int = 2


@app.post("/rag")
async def rag_endpoint(payload: QueryRequest):
    """Legacy RAG endpoint (backward compatible with Task2.py)."""
    # Use the new agent internally
    agent_request = AgentRequest(query=payload.query, k=payload.k)
    response = await agent_endpoint(agent_request)
    return {"query": payload.query, "result": response.narrative}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
