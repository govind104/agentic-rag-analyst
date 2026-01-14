"""
AI Analyst Agent - Integration Tests
Run 10 test queries to validate the system.
"""

import sys
import os
import time
import json

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
def test_data_layer():
    """Test data.py functionality."""
    print("\n=== Test 1: Data Layer ===")
    from data import run_sql, get_table_info, init_database
    
    # Ensure DB exists
    init_database()
    
    # Test queries
    tests = [
        ("SELECT COUNT(*) as count FROM trips", "trips count"),
        ("SELECT COUNT(*) as count FROM customers", "customers count"),
        ("SELECT location, SUM(fare) as total FROM trips GROUP BY location ORDER BY total DESC LIMIT 5", "top 5 locations"),
        ("SELECT region, AVG(churn)*100 as rate FROM customers GROUP BY region", "churn by region"),
    ]
    
    all_passed = True
    for sql, desc in tests:
        try:
            result = run_sql(sql)
            if len(result) > 0:
                print(f"  ✅ {desc}: {len(result)} rows")
            else:
                print(f"  ❌ {desc}: Empty result")
                all_passed = False
        except Exception as e:
            print(f"  ❌ {desc}: {e}")
            all_passed = False
    
    return all_passed


def test_ethics_module():
    """Test ethics.py functionality."""
    print("\n=== Test 2: Ethics Module ===")
    from ethics import check_bias, is_safe_query, redact_sensitive
    
    # Bias tests
    tests = [
        ("The data shows trends.", 0.3, "neutral text"),
        ("He is a strong leader.", 1.0, "gendered text"),  # 1.0 expected due to stereotype words
    ]
    
    all_passed = True
    for text, max_bias, desc in tests:
        result = check_bias(text)
        if result["bias_score"] <= max_bias:
            print(f"  ✅ {desc}: bias={result['bias_score']}")
        else:
            print(f"  ❌ {desc}: bias={result['bias_score']} > {max_bias}")
            all_passed = False
    
    # Safety tests
    safe_queries = [
        ("Top 5 locations", True, "safe query"),
        ("DROP TABLE trips", False, "SQL injection"),
        ("Show passwords", False, "blocked topic"),
    ]
    
    for query, expected_safe, desc in safe_queries:
        result = is_safe_query(query)
        if result == expected_safe:
            print(f"  ✅ {desc}: safe={result}")
        else:
            print(f"  ❌ {desc}: expected safe={expected_safe}, got {result}")
            all_passed = False
    
    # PII redaction
    text = "Email: test@email.com, Phone: 555-123-4567"
    redacted = redact_sensitive(text)
    if "[EMAIL]" in redacted and "[PHONE]" in redacted:
        print(f"  ✅ PII redaction: {redacted}")
    else:
        print(f"  ❌ PII redaction failed: {redacted}")
        all_passed = False
    
    return all_passed


def test_agent_tools():
    """Test agent tools directly."""
    print("\n=== Test 3: Agent Tools ===")
    
    from agent import SQLQueryTool, RetrieveTool, VizTool, BiasTool, cpu_top_k_retrieval
    import numpy as np
    
    all_passed = True
    
    # SQL Tool
    result = SQLQueryTool.run("SELECT location, COUNT(*) as cnt FROM trips GROUP BY location LIMIT 3")
    if result["data"] and len(result["data"]) > 0:
        print(f"  ✅ SQLQueryTool: {len(result['data'])} rows")
    else:
        print(f"  ❌ SQLQueryTool: {result.get('error', 'No data')}")
        all_passed = False
    
    # SQL injection prevention
    result = SQLQueryTool.run("DROP TABLE trips")
    if result["error"] and "Forbidden" in result["error"]:
        print(f"  ✅ SQL injection blocked")
    else:
        print(f"  ❌ SQL injection not blocked")
        all_passed = False
    
    # Viz Tool
    data = [{"x": 1, "y": 10}, {"x": 2, "y": 20}]
    result = VizTool.run(data)
    if result["plotly_json"]:
        print(f"  ✅ VizTool: generated plotly spec")
    else:
        print(f"  ❌ VizTool: {result.get('error', 'No viz')}")
        all_passed = False
    
    # Bias Tool
    result = BiasTool.run("The manager reviewed data.")
    if "bias_score" in result:
        print(f"  ✅ BiasTool: bias={result['bias_score']}")
    else:
        print(f"  ❌ BiasTool failed")
        all_passed = False
    
    # CPU top-K retrieval
    query = np.random.randn(1, 10).astype(np.float32)
    docs = np.random.randn(100, 10).astype(np.float32)
    indices = cpu_top_k_retrieval(query, docs, k=5)
    if len(indices) == 5:
        print(f"  ✅ cpu_top_k_retrieval: returned {len(indices)} indices")
    else:
        print(f"  ❌ cpu_top_k_retrieval: expected 5, got {len(indices)}")
        all_passed = False
    
    return all_passed


def test_sample_queries():
    """Test sample analytical queries."""
    print("\n=== Test 4: Sample Queries ===")
    from data import run_sql
    
    queries = [
        "SELECT location, SUM(fare) as total FROM trips GROUP BY location ORDER BY total DESC LIMIT 5",
        "SELECT passengers, AVG(fare) as avg_fare FROM trips GROUP BY passengers",
        "SELECT region, AVG(churn)*100 as churn_rate FROM customers GROUP BY region",
        "SELECT region, AVG(revenue) as avg_revenue FROM customers GROUP BY region",
        "SELECT strftime('%Y-%m', pickup_date) as month, COUNT(*) as cnt FROM trips GROUP BY month",
    ]
    
    all_passed = True
    for i, sql in enumerate(queries, 1):
        try:
            result = run_sql(sql)
            if len(result) > 0:
                print(f"  ✅ Query {i}: {len(result)} rows")
            else:
                print(f"  ❌ Query {i}: Empty")
                all_passed = False
        except Exception as e:
            print(f"  ❌ Query {i}: {e}")
            all_passed = False
    
    return all_passed


def test_streamlit_imports():
    """Test Streamlit app imports."""
    print("\n=== Test 5: Streamlit Imports ===")
    try:
        import streamlit
        import httpx
        import plotly.graph_objects
        print("  ✅ All Streamlit dependencies imported")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_mlflow_imports():
    """Test MLflow imports."""
    print("\n=== Test 6: MLflow Imports ===")
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        print("  ✅ MLflow imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_distance_metrics():
    """Test CPU distance functions."""
    print("\n=== Test 7: Distance Metrics ===")
    from agent import cpu_top_k_retrieval
    import numpy as np
    
    query = np.random.randn(1, 50).astype(np.float32)
    docs = np.random.randn(100, 50).astype(np.float32)
    
    metrics = ["l2", "cosine", "dot", "manhattan"]
    all_passed = True
    
    for metric in metrics:
        try:
            indices = cpu_top_k_retrieval(query, docs, k=5, metric=metric)
            if len(indices) == 5:
                print(f"  ✅ {metric}: top-5 returned")
            else:
                print(f"  ❌ {metric}: expected 5, got {len(indices)}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ {metric}: {e}")
            all_passed = False
    
    return all_passed


def test_edge_cases():
    """Test edge cases."""
    print("\n=== Test 8: Edge Cases ===")
    from agent import SQLQueryTool, BiasTool
    from ethics import ContentGuardrails
    
    all_passed = True
    
    # Empty query
    result = SQLQueryTool.run("")
    if result["error"]:
        print("  ✅ Empty SQL handled")
    else:
        print("  ❌ Empty SQL not handled")
        all_passed = False
    
    # Empty text for bias
    result = BiasTool.run("")
    if result["bias_score"] == 0.0:
        print("  ✅ Empty text bias handled")
    else:
        print("  ❌ Empty text not handled")
        all_passed = False
    
    # High K value
    result = SQLQueryTool.run("SELECT * FROM trips LIMIT 1000")
    if result["data"]:
        print(f"  ✅ High K query: {len(result['data'])} rows")
    else:
        print("  ❌ High K query failed")
        all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("AI Analyst Agent - Integration Tests")
    print("=" * 60)
    
    start = time.time()
    
    results = {
        "Data Layer": test_data_layer(),
        "Ethics Module": test_ethics_module(),
        "Agent Tools": test_agent_tools(),
        "Sample Queries": test_sample_queries(),
        "Streamlit Imports": test_streamlit_imports(),
        "MLflow Imports": test_mlflow_imports(),
        "Distance Metrics": test_distance_metrics(),
        "Edge Cases": test_edge_cases(),
    }
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Time: {elapsed:.2f}s")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
