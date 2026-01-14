"""
AI Analyst Agent - Streamlit Frontend
Chat UI with sidebar config, live metrics dashboard, and MLflow integration.
"""

import streamlit as st
import httpx
import json
import time
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AI Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Custom CSS for Premium Design
# ==============================================================================
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102,126,234,0.15);
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(0,0,0,0.3) !important;
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Configuration
# ==============================================================================
API_BASE_URL = "http://127.0.0.1:8001"
MLFLOW_URL = "http://localhost:5000"

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = {
        "timestamps": [],
        "latencies": [],
        "bias_scores": []
    }

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

if "viz_counter" not in st.session_state:
    st.session_state.viz_counter = 0

# ==============================================================================
# Sidebar Configuration
# ==============================================================================
with st.sidebar:
    st.markdown("# ⚙️ Configuration")
    st.markdown("---")
    
    # Model settings
    st.markdown("### 🧠 Model Settings")
    
    k_value = st.slider(
        "Top-K Documents",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of documents to retrieve for context"
    )
    
    batch_size = st.slider(
        "Batch Size",
        min_value=4,
        max_value=16,
        value=8,
        help="Batch size for processing"
    )
    
    distance_metric = st.selectbox(
        "Distance Metric",
        options=["l2", "cosine", "dot", "manhattan"],
        index=0,
        help="Metric for similarity search"
    )
    
    st.markdown("---")
    
    # Quick links
    st.markdown("### 🔗 Quick Links")
    
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("📊 MLflow", MLFLOW_URL, width="stretch")
    with col2:
        st.link_button("📖 API Docs", f"{API_BASE_URL}/docs", width="stretch")
    
    st.markdown("---")
    
    # Session info
    st.markdown("### 📝 Session Info")
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    if st.button("🗑️ Clear Chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{int(time.time())}"
        st.rerun()

# ==============================================================================
# Helper Functions
# ==============================================================================
def call_agent(query: str, k: int, metric: str) -> dict:
    """Call the AI Analyst Agent API."""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_BASE_URL}/agent",
                json={
                    "query": query,
                    "k": k,
                    "metric": metric,
                    "session_id": st.session_state.session_id
                }
            )
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to API. Please start the backend server (python agent.py)"}
    except Exception as e:
        return {"error": str(e)}


def render_visualization(viz_code: str, key_suffix: str = ""):
    """Render Plotly visualization from JSON spec."""
    try:
        spec = json.loads(viz_code)
        if spec and "data" in spec:
            fig = go.Figure(data=spec["data"], layout=spec.get("layout", {}))
            st.session_state.viz_counter += 1
            unique_key = f"viz_{st.session_state.viz_counter}_{key_suffix}_{hash(viz_code) % 10000}"
            st.plotly_chart(fig, width="stretch", key=unique_key)
    except Exception as e:
        st.warning(f"Could not render visualization: {e}")


# ==============================================================================
# Main Content Area
# ==============================================================================
# Page navigation
page = st.radio(
    "Navigation",
    ["💬 Chat", "📊 Dashboard", "📚 Documentation"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ==============================================================================
# Chat Page
# ==============================================================================
if page == "💬 Chat":
    st.markdown("# 🤖 AI Analyst Agent")
    st.markdown("*Ask questions about NYC Taxi trips or Customer Churn data*")
    
    # Sample queries
    with st.expander("💡 Sample Queries", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **NYC Taxi Data:**
            - "Top 5 locations by fare"
            - "Average fare by passenger count"
            - "Trips by month"
            """)
        with col2:
            st.markdown("""
            **Customer Churn:**
            - "Churn rate by region"
            - "Average revenue by region"
            - "Tenure vs churn analysis"
            """)
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show additional data for assistant messages
            if message["role"] == "assistant" and "data" in message:
                data = message["data"]
                
                # SQL Query
                if data.get("sql_query"):
                    with st.expander("🔍 SQL Query"):
                        st.code(data["sql_query"], language="sql")
                
                # Visualization
                if data.get("viz_code"):
                    with st.expander("📊 Visualization", expanded=True):
                        render_visualization(data["viz_code"], f"history_{idx}")
                
                # Metrics
                cols = st.columns(3)
                with cols[0]:
                    st.metric("⏱️ Latency", f"{data.get('latency_ms', 0):.0f}ms")
                with cols[1]:
                    st.metric("⚖️ Bias Score", f"{data.get('bias_score', 0):.4f}")
                with cols[2]:
                    st.metric("📄 Session", data.get("session_id", "N/A")[-8:])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🔄 Analyzing..."):
                response = call_agent(prompt, k_value, distance_metric)
            
            if "error" in response:
                st.error(f"❌ Error: {response['error']}")
                narrative = f"Error: {response['error']}"
                data = {}
            else:
                narrative = response.get("narrative", "No response generated.")
                st.markdown(narrative)
                
                data = {
                    "sql_query": response.get("sql_query"),
                    "sql_result": response.get("sql_result"),
                    "viz_code": response.get("viz_code"),
                    "latency_ms": response.get("latency_ms", 0),
                    "bias_score": response.get("bias_score", 0),
                    "session_id": response.get("session_id", "")
                }
                
                # Show SQL
                if data.get("sql_query"):
                    with st.expander("🔍 SQL Query"):
                        st.code(data["sql_query"], language="sql")
                
                # Show visualization
                if data.get("viz_code"):
                    with st.expander("📊 Visualization", expanded=True):
                        render_visualization(data["viz_code"], f"new_{int(time.time())}")
                
                # Show metrics
                cols = st.columns(3)
                with cols[0]:
                    st.metric("⏱️ Latency", f"{data.get('latency_ms', 0):.0f}ms")
                with cols[1]:
                    st.metric("⚖️ Bias Score", f"{data.get('bias_score', 0):.4f}")
                with cols[2]:
                    st.metric("📄 Session", data.get("session_id", "N/A")[-8:])
                
                # Update metrics history
                st.session_state.metrics_history["timestamps"].append(datetime.now())
                st.session_state.metrics_history["latencies"].append(data.get("latency_ms", 0))
                st.session_state.metrics_history["bias_scores"].append(data.get("bias_score", 0))
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": narrative,
            "data": data
        })

# ==============================================================================
# Dashboard Page
# ==============================================================================
elif page == "📊 Dashboard":
    st.markdown("# 📊 Live Metrics Dashboard")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_queries = len(st.session_state.metrics_history["timestamps"])
        st.metric("Total Queries", total_queries)
    
    with col2:
        avg_latency = (
            sum(st.session_state.metrics_history["latencies"]) / 
            max(1, len(st.session_state.metrics_history["latencies"]))
        )
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    with col3:
        avg_bias = (
            sum(st.session_state.metrics_history["bias_scores"]) / 
            max(1, len(st.session_state.metrics_history["bias_scores"]))
        )
        st.metric("Avg Bias Score", f"{avg_bias:.4f}")
    
    with col4:
        st.metric("Session ID", st.session_state.session_id[-8:])
    
    st.markdown("---")
    
    # Charts
    if len(st.session_state.metrics_history["timestamps"]) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⏱️ Latency Over Time")
            fig_latency = go.Figure()
            fig_latency.add_trace(go.Scatter(
                x=list(range(len(st.session_state.metrics_history["latencies"]))),
                y=st.session_state.metrics_history["latencies"],
                mode='lines+markers',
                name='Latency (ms)',
                line=dict(color='#667eea', width=2),
                marker=dict(size=8)
            ))
            fig_latency.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                xaxis_title="Query #",
                yaxis_title="Latency (ms)"
            )
            st.plotly_chart(fig_latency, width="stretch", key="dashboard_latency")
        
        with col2:
            st.markdown("### ⚖️ Bias Score Over Time")
            fig_bias = go.Figure()
            fig_bias.add_trace(go.Scatter(
                x=list(range(len(st.session_state.metrics_history["bias_scores"]))),
                y=st.session_state.metrics_history["bias_scores"],
                mode='lines+markers',
                name='Bias Score',
                line=dict(color='#764ba2', width=2),
                marker=dict(size=8)
            ))
            fig_bias.add_hline(y=0.05, line_dash="dash", line_color="red", 
                             annotation_text="Threshold (0.05)")
            fig_bias.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                xaxis_title="Query #",
                yaxis_title="Bias Score"
            )
            st.plotly_chart(fig_bias, width="stretch", key="dashboard_bias")
    else:
        st.info("📈 No metrics yet. Start chatting to see live metrics!")
    
    # Recent queries table
    st.markdown("### 📝 Recent Queries")
    if st.session_state.messages:
        queries_data = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                queries_data.append({
                    "Query": msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"],
                    "Type": "User"
                })
        if queries_data:
            st.dataframe(pd.DataFrame(queries_data), width="stretch")
    else:
        st.info("No queries yet.")

# ==============================================================================
# Documentation Page
# ==============================================================================
elif page == "📚 Documentation":
    st.markdown("# 📚 Documentation")
    
    st.markdown("""
    ## 🤖 AI Analyst Agent
    
    A production-ready GenAI demo showcasing:
    - **Agentic RAG** with LangGraph state machine
    - **SQL Query Generation** for NYC Taxi and Customer Churn data
    - **Automated Visualization** with Plotly
    - **Bias Detection** for ethical AI
    - **MLflow Tracking** for experiment management
    
    ---
    
    ## 🗄️ Available Data
    
    ### NYC Taxi Trips
    | Column | Type | Description |
    |--------|------|-------------|
    | id | INTEGER | Trip ID |
    | pickup_date | TIMESTAMP | Pickup datetime |
    | location | INTEGER | NYC taxi zone (1-265) |
    | fare | FLOAT | Trip fare in USD |
    | passengers | INTEGER | Passenger count |
    
    ### Customer Churn
    | Column | Type | Description |
    |--------|------|-------------|
    | id | INTEGER | Customer ID |
    | region | TEXT | Geographic region |
    | tenure | INTEGER | Months as customer |
    | churn | INTEGER | Churned (1) or not (0) |
    | revenue | FLOAT | Customer revenue in USD |
    
    ---
    
    ## 🔧 API Endpoints
    
    | Endpoint | Method | Description |
    |----------|--------|-------------|
    | `/agent` | POST | Main agent endpoint |
    | `/rag` | POST | Legacy RAG endpoint |
    | `/metrics` | GET | Prometheus metrics |
    | `/health` | GET | Health check |
    | `/tables` | GET | Database schema |
    
    ---
    
    ## 🏃 Running Locally
    
    ```bash
    # Start backend
    python agent.py
    
    # Start frontend (new terminal)
    streamlit run app.py
    
    # Start MLflow (optional)
    ./mlflow_run.sh
    ```
    
    ---
    
    ## 🐳 Docker Deployment
    
    ```bash
    docker-compose up --build
    ```
    
    This starts:
    - Streamlit UI on port 8501
    - FastAPI backend on port 8001
    - MLflow on port 5000
    """)

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.5);'>"
    "🤖 AI Analyst Agent | Built with Streamlit, FastAPI, LangGraph & MLflow"
    "</div>",
    unsafe_allow_html=True
)
