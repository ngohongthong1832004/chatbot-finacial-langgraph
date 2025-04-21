import psycopg2
import pandas as pd

import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()


# Dùng os.environ để truy cập
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def get_db_config():
    return {
    "dbname": os.getenv("DBNAME"),
    "user": os.getenv("DBUSER"),
    "password": os.getenv("DBPASSWORD"),
    "host": os.getenv("DBHOST"),
    "port": int(os.getenv("DBPORT", 5432)),
    "sslmode": os.getenv("SSL_MODE", "require")
}

def connect_to_database():
    try:
        conn = psycopg2.connect(**get_db_config())
        print("✅ Successfully connected to PostgreSQL.")
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def execute_sql_query(conn, query):
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"❌ SQL execution error: {e}")
        return None

def get_schema_and_samples(conn):
    try:
        # Get all tables
        tables_query = """
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public'
"""
        tables_df = pd.read_sql(tables_query, conn)
        
        schema_info = {}
        
        for table in tables_df['table_name']:
            # Get columns for this table
            columns_query = f"""
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = '{table}'
"""
            columns_df = pd.read_sql(columns_query, conn)
            
            # Get sample data (first 5 rows)
            sample_query = f"SELECT * FROM {table} LIMIT 5"
            try:
                sample_df = pd.read_sql(sample_query, conn)
                sample_data = sample_df.to_dict(orient='records')
            except:
                sample_data = []
            
            schema_info[table] = {
                'columns': columns_df.to_dict(orient='records'),
                'sample_data': sample_data
            }
        
        return schema_info
        
    except Exception as e:
        print(f"❌ Error getting schema info: {e}")
        return {"error": str(e)}

def generate_sql_query(question, schema_info=None):
    """Generate SQL query from natural language question using LLM"""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)
    
    # If schema_info is not provided, use a generic approach
    if not schema_info:
        schema_description = "Unknown schema. Try to generate a generic SQL query."
    else:
        # Format schema info into a readable string
        schema_parts = []
        for table_name, table_info in schema_info.items():
            columns = ", ".join([f"{col['column_name']} ({col['data_type']})" 
                               for col in table_info['columns']])
            schema_parts.append(f"Table: {table_name}\nColumns: {columns}")
            
            # Add sample data if available
            if table_info['sample_data']:
                sample = str(table_info['sample_data'][0])
                schema_parts.append(f"Sample row: {sample}")
                
        schema_description = "\n\n".join(schema_parts)
    
    prompt = ChatPromptTemplate.from_template("""
You are an expert SQL developer. Generate a SQL query to answer the user's question.
Use the following database schema information:

{schema}

User's question: {question}

Return ONLY the SQL query without any explanation or markdown formatting.
Make sure the query is correct PostgreSQL syntax.
"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    sql_query = chain.invoke({
        "schema": schema_description,
        "question": question
    })
    
    return sql_query 