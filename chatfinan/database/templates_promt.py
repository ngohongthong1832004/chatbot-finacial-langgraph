# sql
TABLES_QUERY= """
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public'
"""
GET_COLUMNS_BY_TABLE  = """
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_schema = 'public' 
AND table_name = '{table}'
"""
SAMPLE_QUERY= "SELECT * FROM {table} LIMIT 5"
PROMT_SQL = """
You are an expert SQL developer. Generate a SQL query to answer the user's question.
Use the following database schema information:

{schema}

User's question: {question}

Return ONLY the SQL query without any explanation or markdown formatting.
Make sure the query is correct PostgreSQL syntax.
"""