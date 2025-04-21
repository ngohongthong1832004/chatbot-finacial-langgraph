from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.database.connection import connect_to_database, execute_sql_query, get_schema_and_samples, generate_sql_query

import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Dùng os.environ để truy cập
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

DB_CONFIG = {
    "dbname": os.getenv("DBNAME"),
    "user": os.getenv("DBUSER"),
    "password": os.getenv("DBPASSWORD"),
    "host": os.getenv("DBHOST"),
    "port": int(os.getenv("DBPORT", 5432)),
    "sslmode": os.getenv("SSL_MODE", "require")
}


chroma_db = None
similarity_threshold_retriever = None



# Data model for graph state
class GraphState(BaseModel):
    """State for the RAG graph"""
    question: str = Field(..., description="The user's question")
    documents: List[Document] = Field(default_factory=list, description="List of retrieved documents")
    web_search_needed: str = Field(default="No", description="Whether web search is needed")
    use_sql: str = Field(default="No", description="Whether to use SQL query")
    generation: str = Field(default="", description="The generated answer")
    
    def __str__(self):
        return f"Question: {self.question}\nDocs: {len(self.documents)}\nUse SQL: {self.use_sql}, WebSearch: {self.web_search_needed}\nGeneration: {self.generation[:80]}..."

# Initialize models and tools
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small', api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)
chatgpt = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000, api_key=TAVILY_API_KEY)

# Initialize vector store with fallback & debug
def init_vector_store():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # vị trí thực tế của file này
        persist_path = os.path.abspath(os.path.join(BASE_DIR, "../../wikipedia_db"))
        print(f"📁 Loading vector DB from: {persist_path}")
        
        db = Chroma(
            collection_name='rag_wikipedia_db',
            embedding_function=openai_embed_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=persist_path
        )
        print(f"✅ Vector DB loaded with {db._collection.count()} documents.")
        return db
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize vector store. Error: {e}")
        return None
    
def get_vector_store_and_retriever():
    global chroma_db, similarity_threshold_retriever
    if chroma_db is not None and similarity_threshold_retriever is not None:
        print("✅ Vector store & retriever already initialized. Reusing...")
        return chroma_db, similarity_threshold_retriever

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        persist_path = os.path.abspath(os.path.join(BASE_DIR, "../../wikipedia_db"))
        print(f"📁 Loading vector DB from: {persist_path}")
        
        chroma_db = Chroma(
            collection_name='rag_wikipedia_db',
            embedding_function=openai_embed_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=persist_path
        )
        print(f"✅ Vector DB loaded with {chroma_db._collection.count()} documents.")
        
        try:
            similarity_threshold_retriever = chroma_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.3}
            )
            print("✅ Retriever initialized with similarity_score_threshold.")
        except Exception as e:
            print(f"⚠️ Failed to init threshold-based retriever: {e}")
            similarity_threshold_retriever = chroma_db.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )
            print("🔁 Fallback to regular similarity retriever.")

    except Exception as e:
        print(f"❌ Failed to initialize Chroma DB: {e}")
        chroma_db = None
        similarity_threshold_retriever = None

    return chroma_db, similarity_threshold_retriever


# Create grader chain
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert grader assessing relevance of a retrieved document to a user question.\nAnswer only 'yes' or 'no' depending on whether the document is relevant to the question."),
    ("human", "Retrieved document:\n{document}\n\nUser question:\n{question}")
])
doc_grader = grade_prompt | llm | StrOutputParser()

# Create QA chain
prompt_template = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If no context is present or if you don't know the answer, just say that you don't know the answer.
    Do not make up the answer unless it is there in the provided context.
    Give a detailed answer and to the point answer with regard to the question.
    Question:
    {question}
    Context:
    {context}
    Answer:
    """
)

def format_docs(docs):
    return "\n\n".join(doc.page_content if isinstance(doc, Document) else str(doc) for doc in docs)

qa_rag_chain = (
    {
        "context": (itemgetter('context') | RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
    | prompt_template 
    | chatgpt
    | StrOutputParser()
)

# Query rewriter for web search
re_write_prompt = ChatPromptTemplate.from_template(
    """Given a user question, rewrite it to be a more effective web search query.
Original question: {question}
Search query:"""
)
question_rewriter = (re_write_prompt|llm|StrOutputParser())

# Graph components
def retrieve(state):
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state.question
    documents = []

    _, retriever = get_vector_store_and_retriever()

    if retriever:
        try:
            documents = retriever.invoke(question)
            # print("Documents :",documents)
            # print(f"Retrieved {len(documents)} documents")
        except Exception as e:
            print(f"Error during retrieval: {e}")
    else:
        print("❌ No retriever available.")

    return {"documents": documents, "question": question}

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state.question
    documents = state.documents
    
    filtered_docs = []
    web_search_needed = "No"
    use_sql = "No"
    
    # Define database related keywords
    database_keywords = [
        "database", 
        "sql",
        "select",
        "from table",
        "stocks", # specific table name
        "query",
        "table",
        "dữ liệu",
        "cơ sở dữ liệu"
    ]

    # Check if question is related to database first
    if any(keyword in question.lower() for keyword in database_keywords):
        use_sql = "Yes"
        print("---GRADE: DETECTED DATABASE QUERY---")
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search_needed": "No",
            "use_sql": "Yes"
        }
    
    if documents:
        for d in documents:
            score = doc_grader.invoke({
                "question": question,
                "document": d.page_content
            })
            if score.strip().lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT, SUGGEST WEB SEARCH---")
                web_search_needed = "Yes"
    else:
        print("---NO DOCUMENTS RETRIEVED---")
        web_search_needed = "Yes"
    
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search_needed": web_search_needed,
        "use_sql": use_sql
    }

def rewrite_query(state):
    print("---REWRITING QUERY FOR WEB SEARCH---")
    question = state.question
    documents = state.documents
    
    rewritten_question = question_rewriter.invoke({"question": question})
    print(f"Rewritten question: {rewritten_question}")
    
    return {"documents": documents, "question": rewritten_question, "original_question": question}

def web_search(state):
    print("---WEB SEARCH---")
    question = getattr(state, "question")
    original_question = getattr(state, "original_question", question)
    documents = state.documents
    
    try:
        search_results = tv_search.invoke(question)
        web_content = "\n\n".join([f"Source: {res['url']}\n{res['content']}" for res in search_results])
        web_doc = Document(page_content=web_content)
        documents.append(web_doc)
        print(f"Added web search results ({len(search_results)} sources)")
    except Exception as e:
        print(f"Error during web search: {e}")
    
    return {"documents": documents, "question": original_question}

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

        # Execute query
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
        # If SQL fails, try web search as fallback
        return {
            "documents": state.documents + [Document(page_content=f"Error: {error_msg}")],
            "question": question,
            "web_search_needed": "Yes",
            "use_sql": "No"
        }

def generate_answer(state):
    print("---GENERATE ANSWER---")
    question = state.question
    documents = state.documents
    use_sql = state.use_sql

    if not documents:
        generation = "I don't have enough information to answer this question."
    elif use_sql == "Yes":
    # Tách riêng document chứa SQL và document chứa kết quả
        sql_doc = next((doc for doc in documents if doc.page_content.startswith("SQL used:")), None)
        result_doc = next((doc for doc in documents if doc.page_content.startswith("SQL Query Results:")), None)

        generation_parts = []
        if sql_doc:
            generation_parts.append(sql_doc.page_content)
        if result_doc:
            generation_parts.append(result_doc.page_content)

        generation = "\n\n".join(generation_parts) if generation_parts else "No SQL content found."
    else:
        # Loại bỏ documents trùng lặp
        unique_docs = list({doc.page_content: doc for doc in documents}.values())
        formatted_context = format_docs(unique_docs)

        # print("---FORMATTED CONTEXT PREVIEW---")
        # print(formatted_context[:300])

        generation = qa_rag_chain.invoke({
            "context": formatted_context,
            "question": question
        })

    # print(f"Question: {question}")
    # print(f"documents: {formatted_context}")
    # print(f"Answer: {generation}")
    # print(f"Documents count: {len(documents)}")

    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = getattr(state, "web_search_needed", "No")
    use_sql = getattr(state, "use_sql", "No")
    
    if use_sql == "Yes":
        print("---DECISION: QUERY SQL DATABASE---")
        return "query_sql"
    elif web_search_needed == "Yes":
        print("---DECISION: WEB SEARCH NEEDED, REWRITE QUERY---")
        return "rewrite_query"
    else:
        print("---DECISION: DOCUMENTS ARE RELEVANT, GENERATE ANSWER---")
        return "generate_answer"

# Initialize graph

def create_rag_graph():
    agentic_rag = StateGraph(GraphState)

    # Nodes
    agentic_rag.add_node("retrieve", retrieve)
    agentic_rag.add_node("grade_documents", grade_documents)
    agentic_rag.add_node("rewrite_query", rewrite_query)
    agentic_rag.add_node("web_search", web_search)
    agentic_rag.add_node("query_sql", query_sql)
    agentic_rag.add_node("generate_answer", generate_answer)

    # ✅ Entry point
    agentic_rag.set_entry_point("retrieve")

    # Edges
    agentic_rag.add_edge("retrieve", "grade_documents")
    agentic_rag.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
            "query_sql": "query_sql"
        }
    )
    agentic_rag.add_edge("rewrite_query", "web_search")
    agentic_rag.add_edge("web_search", "generate_answer")
    agentic_rag.add_edge("query_sql", "generate_answer")
    agentic_rag.add_edge("generate_answer", END)

    return agentic_rag.compile()


def process_query(query: str) -> Dict[str, Any]:
    """
    Gọi pipeline Agentic RAG để xử lý câu hỏi và trả về kết quả.
    """
    print(f"📥 Processing query: {query}")
    rag_graph = create_rag_graph()
    result = rag_graph.invoke({"question": query})
    print(f"📤 Done. Generation: {result.get('generation', '')[:100]}")
    return result
