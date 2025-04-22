"""
SQL Agent Implementation
This file provides a complete implementation of a SQL agent using LangChain.
It includes setup, usage examples, and documentation.
"""

# Required imports
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd

# Load environment variables
load_dotenv()

class SQLAgent:
    def __init__(self, db_uri):
        """
        Initialize SQL Agent with database connection
        
        Args:
            db_uri (str): Database connection URI (e.g., "sqlite:///./data/example.db")
        """
        # Initialize SQL database connection
        self.db = SQLDatabase.from_uri(db_uri)
        
        # Create SQL agent
        self.agent = create_sql_agent(
            llm=ChatOpenAI(temperature=0, model="gpt-4"),
            db=self.db,
            agent_type="openai-tools",
            verbose=True
        )
    
    def query(self, question):
        """
        Execute a natural language query on the database
        
        Args:
            question (str): Natural language question about the database
            
        Returns:
            str: Query result in human-readable format
        """
        try:
            result = self.agent.invoke({"input": question})
            return result
        except Exception as e:
            return f"Error executing query: {str(e)}"

def setup_example_database():
    """
    Creates an example SQLite database with sample data
    """
    # Create database directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect("./data/example.db")
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        age INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create orders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        product_name TEXT NOT NULL,
        amount REAL NOT NULL,
        order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    
    # Insert sample data
    users_data = [
        (1, "John Doe", "john@example.com", 30),
        (2, "Jane Smith", "jane@example.com", 25),
        (3, "Bob Johnson", "bob@example.com", 35)
    ]
    
    orders_data = [
        (1, 1, "Laptop", 999.99),
        (2, 1, "Mouse", 29.99),
        (3, 2, "Keyboard", 49.99),
        (4, 3, "Monitor", 199.99)
    ]
    
    cursor.executemany("INSERT OR IGNORE INTO users (id, name, email, age) VALUES (?, ?, ?, ?)", users_data)
    cursor.executemany("INSERT OR IGNORE INTO orders (id, user_id, product_name, amount) VALUES (?, ?, ?, ?)", orders_data)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()

def main():
    """
    Main function demonstrating SQL agent usage
    """
    # Setup example database
    print("Setting up example database...")
    setup_example_database()
    
    # Initialize SQL agent
    print("\nInitializing SQL agent...")
    sql_agent = SQLAgent("sqlite:///./data/example.db")
    
    # Example queries
    queries = [
        "What is the total number of users?",
        "What is the average age of users?",
        "Show me all orders with their user information",
        "What is the total amount spent by each user?",
        "Who made the most expensive purchase?"
    ]
    
    # Execute queries
    print("\nExecuting example queries:")
    for query in queries:
        print(f"\nQuery: {query}")
        result = sql_agent.query(query)
        print(f"Result: {result}")

if __name__ == "__main__":
    main() 