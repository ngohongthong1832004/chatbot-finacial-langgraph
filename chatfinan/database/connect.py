import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

from langchain_core.output_parsers import StrOutputParser
from .templates_promt import TABLES_QUERY, GET_COLUMNS_BY_TABLE, SAMPLE_QUERY, PROMT_SQL

load_dotenv()
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
        # get all tables
        tables_df = pd.read_sql(TABLES_QUERY, conn)
        schema_info = {}
        
        for table in tables_df['table_name']:
            # Get columns 
            columns_query = GET_COLUMNS_BY_TABLE.format(table=table)
            columns_df = pd.read_sql(columns_query, conn)
            sample_query = SAMPLE_QUERY.format(table=table) 
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
    """Generate SQL query """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    if not schema_info:
        schema_description = "Unknown schema. Try to generate a generic SQL query."
    else:
        # Process data schema
        schema_parts = []
        for table_name, table_info in schema_info.items():
            columns = ", ".join([f"{col['column_name']} ({col['data_type']})" 
                               for col in table_info['columns']])
            schema_parts.append(f"Table: {table_name}\nColumns: {columns}")
            if table_info['sample_data']:
                sample = str(table_info['sample_data'][0])
                schema_parts.append(f"Sample row: {sample}")
                
        schema_description = "\n\n".join(schema_parts)
    
    prompt = ChatPromptTemplate.from_template(PROMT_SQL)
    chain = prompt | llm | StrOutputParser()
    
    sql_query = chain.invoke({
        "schema": schema_description,
        "question": question
    })
    return sql_query 

def query_sql(state):
    print("---EXECUTE SQL QUERY---")
    question = state.question
    
    try:
        # Connect to database
        conn = connect_to_database()
        if conn is None:
            error_msg = "Cannot connect to database"
            print(f"❌ {error_msg}")
            return {
                "documents": state.documents + [Document(page_content=f"Error: {error_msg}")],
                "question": question,
                "web_search_needed": "Yes",
                "use_sql": "No"
            }

        # Get schema info
        schema_info = get_schema_and_samples(conn)
        if not schema_info or "error" in schema_info:
            error_msg = f"Cannot get schema information: {schema_info.get('error', 'Unknown error')}"
            print(f"⚠️ {error_msg}")
            conn.close()
            return {
                "documents": state.documents + [Document(page_content=f"Error: {error_msg}")],
                "question": question,
                "web_search_needed": "Yes",
                "use_sql": "No"
            }

        # Generate SQL query
        sql_query = generate_sql_query(question, schema_info)
        if not sql_query:
            error_msg = "Cannot generate SQL query from question"
            print(f"⚠️ {error_msg}")
            conn.close()
            return {
                "documents": state.documents + [Document(page_content=f"Error: {error_msg}")],
                "question": question,
                "web_search_needed": "Yes",
                "use_sql": "No"
            }

        print(f"🧠 Generated SQL:\n{sql_query}")
        results = execute_sql_query(conn, sql_query)
        conn.close()
        if results is None or results.empty:
            content = "No results found from SQL query"
        else:
            # Format results nicely
            content = f"SQL Query Results:\n{results.to_markdown(index=False)}"
            print("📊 Query Results:\n", content)

        # Create document for next steps
        sql_doc = Document(page_content=f"SQL used:\n{sql_query}")
        result_doc  = Document(page_content=content)
        
        return {
            "documents": state.documents + [sql_doc, result_doc],
            "question": question,
            "web_search_needed": "No",
            "use_sql": "Yes"
        }

    except Exception as e:
        error_msg = f"Error executing SQL query: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "documents": state.documents + [Document(page_content=f"Error: {error_msg}")],
            "question": question,
            "web_search_needed": "Yes",
            "use_sql": "No"
        }
