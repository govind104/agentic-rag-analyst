"""
AI Analyst Agent - Data Layer
SQLite database initialization with NYC Taxi and Customer Churn datasets.
Provides run_sql() function for executing queries.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Database configuration
DB_PATH = Path("data/analyst.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Seed for reproducibility
np.random.seed(42)
random.seed(42)


def get_connection() -> sqlite3.Connection:
    """Get SQLite database connection."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def run_sql(query: str) -> pd.DataFrame:
    """
    Execute SQL query and return results as DataFrame.
    
    Args:
        query: SQL query string
        
    Returns:
        pandas DataFrame with query results
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def generate_nyc_taxi_data(n_rows: int = 10000) -> pd.DataFrame:
    """Generate synthetic NYC Taxi trip data."""
    
    # NYC location IDs (common pickup/dropoff zones)
    location_ids = list(range(1, 266))  # NYC has 265 taxi zones
    
    # Generate dates over last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    data = {
        "id": range(1, n_rows + 1),
        "pickup_date": [
            start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            for _ in range(n_rows)
        ],
        "location": np.random.choice(location_ids, n_rows),
        "fare": np.round(np.random.exponential(15, n_rows) + 2.5, 2),  # Min $2.50
        "passengers": np.random.choice([1, 2, 3, 4, 5, 6], n_rows, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
    }
    
    df = pd.DataFrame(data)
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    return df


def generate_churn_data(n_rows: int = 10000) -> pd.DataFrame:
    """Generate synthetic customer churn data."""
    
    regions = ["North", "South", "East", "West", "Central"]
    
    data = {
        "id": range(1, n_rows + 1),
        "region": np.random.choice(regions, n_rows, p=[0.25, 0.20, 0.20, 0.20, 0.15]),
        "tenure": np.random.randint(1, 73, n_rows),  # 1-72 months
        "churn": np.random.choice([0, 1], n_rows, p=[0.73, 0.27]),  # ~27% churn rate
        "revenue": np.round(np.random.exponential(100, n_rows) + 20, 2)  # Min $20
    }
    
    df = pd.DataFrame(data)
    
    # Make churn more realistic - higher tenure = lower churn
    high_tenure_mask = df["tenure"] > 36
    df.loc[high_tenure_mask, "churn"] = np.random.choice(
        [0, 1], 
        high_tenure_mask.sum(), 
        p=[0.85, 0.15]
    )
    
    return df


def init_database(force_recreate: bool = False) -> None:
    """
    Initialize SQLite database with sample data.
    
    Args:
        force_recreate: If True, drop and recreate tables
    """
    # Delete existing database file if corrupted or if force_recreate
    if force_recreate and DB_PATH.exists():
        print(f"Force recreating database at {DB_PATH}...")
        DB_PATH.unlink()
    
    # Try to initialize, delete and retry on corruption
    try:
        _do_init_database(force_recreate)
    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")
        print(f"Deleting corrupted database at {DB_PATH} and retrying...")
        if DB_PATH.exists():
            DB_PATH.unlink()
        _do_init_database(force_recreate=False)


def _do_init_database(force_recreate: bool = False) -> None:
    """Internal database initialization logic."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        if force_recreate:
            cursor.execute("DROP TABLE IF EXISTS trips")
            cursor.execute("DROP TABLE IF EXISTS customers")
        
        # Create trips table (NYC Taxi)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trips (
                id INTEGER PRIMARY KEY,
                pickup_date TIMESTAMP,
                location INTEGER,
                fare REAL,
                passengers INTEGER
            )
        """)
        
        # Create customers table (Churn)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                region TEXT,
                tenure INTEGER,
                churn INTEGER,
                revenue REAL
            )
        """)
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM trips")
        trips_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM customers")
        customers_count = cursor.fetchone()[0]
        
        # Insert data if tables are empty
        if trips_count == 0:
            print("Generating NYC Taxi data...")
            taxi_df = generate_nyc_taxi_data(10000)
            taxi_df.to_sql("trips", conn, if_exists="replace", index=False)
            print(f"  Inserted {len(taxi_df)} taxi trip records")
        
        if customers_count == 0:
            print("Generating Customer Churn data...")
            churn_df = generate_churn_data(10000)
            churn_df.to_sql("customers", conn, if_exists="replace", index=False)
            print(f"  Inserted {len(churn_df)} customer records")
        
        conn.commit()
        print("Database initialized successfully!")
        
    finally:
        conn.close()


def get_table_info() -> dict:
    """Get information about available tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        info = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [{"name": row[1], "type": row[2]} for row in cursor.fetchall()]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            info[table] = {"columns": columns, "row_count": count}
        
        return info
        
    finally:
        conn.close()


def get_sample_queries() -> list[dict]:
    """Return sample queries for the agent."""
    return [
        {
            "question": "What are the top 5 locations by total fare?",
            "sql": "SELECT location, SUM(fare) as total_fare FROM trips GROUP BY location ORDER BY total_fare DESC LIMIT 5"
        },
        {
            "question": "What is the average fare by passenger count?",
            "sql": "SELECT passengers, AVG(fare) as avg_fare FROM trips GROUP BY passengers ORDER BY passengers"
        },
        {
            "question": "What is the churn rate by region?",
            "sql": "SELECT region, AVG(churn) * 100 as churn_rate FROM customers GROUP BY region ORDER BY churn_rate DESC"
        },
        {
            "question": "What is the average revenue by tenure bucket?",
            "sql": """
                SELECT 
                    CASE 
                        WHEN tenure <= 12 THEN '0-12 months'
                        WHEN tenure <= 24 THEN '13-24 months'
                        WHEN tenure <= 36 THEN '25-36 months'
                        ELSE '37+ months'
                    END as tenure_bucket,
                    AVG(revenue) as avg_revenue,
                    AVG(churn) * 100 as churn_rate
                FROM customers 
                GROUP BY tenure_bucket
                ORDER BY tenure_bucket
            """
        },
        {
            "question": "How many trips were made each month?",
            "sql": "SELECT strftime('%Y-%m', pickup_date) as month, COUNT(*) as trip_count FROM trips GROUP BY month ORDER BY month"
        }
    ]


# Initialize on import
if __name__ == "__main__":
    print("Initializing AI Analyst Agent Database...")
    init_database(force_recreate=True)
    
    print("\nTable Information:")
    for table, info in get_table_info().items():
        print(f"\n{table}:")
        print(f"  Rows: {info['row_count']}")
        print(f"  Columns: {[c['name'] for c in info['columns']]}")
    
    print("\nSample Query Test:")
    result = run_sql("SELECT location, SUM(fare) as total FROM trips GROUP BY location ORDER BY total DESC LIMIT 5")
    print(result)
