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
from langchain_core.messages import HumanMessage
# from vertexai.generative_models import GenerativeModel

import os
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import google.generativeai as genai

import requests
from dotenv import load_dotenv

load_dotenv()

# Dùng os.environ để truy cập
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

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
ENABLE_WEB_SEARCH = True
ENABLE_GPT_GRADING = True


chunks, index, embedding_model = None, None, None
def call_openrouter(prompt_obj) -> str:
    prompt = prompt_obj.to_string()  

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert grader assessing relevance of a retrieved document to a user question. Answer only 'yes' or 'no'."},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"OpenRouter API error: {res.status_code} - {res.text}")


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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY) if ENABLE_GPT_GRADING else None
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000, api_key=TAVILY_API_KEY)


def get_vector_store_and_retriever(resource_dir: str = "sec_embeddings") -> Tuple[List[Dict[str, Any]], faiss.Index, SentenceTransformer]:
    global chunks, index, embedding_model
    
    if chunks is not None and index is not None and embedding_model is not None:
        print("✅ Vector store & retriever already initialized. Reusing...")
        return chunks, index, embedding_model

    try:
        print(f"📁 Loading RAG vector store from: {resource_dir}")
        
        with open(os.path.join(resource_dir, "chunks.json"), 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        with open(os.path.join(resource_dir, "embeddings.pkl"), 'rb') as f:
            embeddings = pickle.load(f)
        
        index = faiss.read_index(os.path.join(resource_dir, "faiss_index.bin"))
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"✅ Loaded {len(chunks)} chunks into memory.")
        print("✅ FAISS index and SentenceTransformer initialized.")

    except Exception as e:
        print(f"❌ Failed to load vector store and retriever: {e}")
        chunks = None
        index = None
        embedding_model = None

    return chunks, index, embedding_model

# Create grader chain
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert grader assessing relevance of a retrieved document to a user question.\n"
               "You will assess both the content and metadata.\n"
               "If the document explains a term or field name explicitly mentioned in the question, answer 'yes'."),
    ("human", "Document:\n{document}\n\nMetadata:\n{metadata}\n\nQuestion:\n{question}")
])
doc_grader = grade_prompt | llm | StrOutputParser()
# doc_grader = grade_prompt | RunnableLambda(call_openrouter) | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content if isinstance(doc, Document) else str(doc) for doc in docs)


def call_gemini_rag(question: str, context: str) -> str:
    prompt = f"""
        Bạn là một trợ lý thông minh có nhiệm vụ trả lời câu hỏi của người dùng.
        Hãy sử dụng những đoạn thông tin dưới đây (được truy xuất từ tài liệu) để trả lời.
        Nếu không có thông tin phù hợp trong đoạn văn, hãy nói rằng bạn không biết câu trả lời.
        Tuyệt đối không tự bịa ra thông tin nếu nó không có trong phần ngữ cảnh.

        Câu hỏi: {question}

        Ngữ cảnh:
        {context}

        Trả lời:
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()




# Query rewriter for web search
re_write_prompt = ChatPromptTemplate.from_template(
    """Hãy viết lại câu hỏi dưới đây sao cho ngắn gọn, rõ ràng và phù hợp để tìm kiếm trên Google.
    
Yêu cầu: Câu hỏi phải hướng tới thông tin mới nhất, cập nhật tại thời điểm hiện tại (không sử dụng năm cũ hoặc dữ liệu đã lỗi thời) chẳng hạn trong năm 2025.

Câu hỏi gốc: {question}
Câu hỏi để tìm kiếm trên web:"""
)
question_rewriter = (re_write_prompt|llm|StrOutputParser())
# question_rewriter = (re_write_prompt|RunnableLambda(call_openrouter)|StrOutputParser())



def retrieve(state):
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state.question
    documents = []

    chunks, index, model = get_vector_store_and_retriever(resource_dir=os.path.join(os.path.dirname(__file__), "sec_embeddings"))

    if index and chunks and model:
        try:
            # Encode query
            query_embedding = model.encode([question])[0].reshape(1, -1).astype(np.float32)

            # Search in the index
            distances, indices = index.search(query_embedding, 3)
            print(f"📌 Retrieved top {len(indices[0])} docs from FAISS")

            for i, idx in enumerate(indices[0]):
                if idx < len(chunks):
                    chunk = chunks[idx]
                    content = chunk["content"]
                    metadata = chunk.get("metadata", {})
                    score = float(1.0 / (1.0 + distances[0][i]))
                    doc = Document(page_content=content, metadata=metadata)
                    doc.metadata["score"] = score
                    documents.append(doc)
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
    else:
        print("❌ No index/model/chunks available.")

    return {"documents": documents, "question": question}

def is_sql_question(question: str) -> bool:
    response = llm.invoke([
        HumanMessage(content=f"""Bạn là chuyên gia phân loại truy vấn. 
Hãy xác định xem câu hỏi sau đây có yêu cầu truy vấn dữ liệu từ cơ sở dữ liệu SQL hay không. 
Trả lời chỉ 'yes' hoặc 'no'.

Câu hỏi: {question}""")
    ])
    return response.content.strip().lower() == "yes"

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state.question
    documents = state.documents

    filtered_docs = []
    web_search_needed = "No"
    use_sql = "No"

    # Check SQL query keywords
    database_keywords = [
        "lấy dữ liệu", "database", "sql", "select", "from table", "stocks", "query", "table", 
        "dữ liệu", "cơ sở dữ liệu", "truy vấn", "truy vấn sql", "lấy dữ liệu", "sinh sql", "câu lệnh"
    ]

    # if any(keyword in question.lower() for keyword in database_keywords):
    #     use_sql = "Yes"
    if is_sql_question(question):
        use_sql = "Yes"
        print("---GRADE: DETECTED DATABASE QUERY---")
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search_needed": "No",
            "use_sql": "Yes"
        }

    # Grade retrieved documents
    if documents:
        for i, d in enumerate(documents):
            print(f"\n📝 Grading document #{i+1}")
            print(f"📌 Question:\n{question}")
            print(f"📄 Document Content (preview):\n{d.page_content[:1000]}")
            print("-" * 60)

            score = doc_grader.invoke({
                "question": question,
                "document": d.page_content,
                "metadata": json.dumps(d.metadata, indent=2)
            })

            if score.strip().lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)

        if not filtered_docs:
            print("---GRADE: NO RELEVANT DOCUMENTS, WEB SEARCH NEEDED---")
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

# def call_openai_rag(question: str) -> str:
#     prompt = f"""
#     Bạn là trợ lý AI chuyên trả lời các câu hỏi tổng hợp kiến thức. Hãy cố gắng trả lời ngắn gọn, chính xác và súc tích.

#     Câu hỏi: {question}

#     Trả lời:
#     """
#     response = llm.invoke([
#         HumanMessage(content=prompt)
#     ])
#     return response.content.strip()

def call_gemini_free_search(question: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")

    try:
        prompt = f"""Hãy trả lời ngắn gọn và chính xác câu hỏi sau, nếu có thể hãy trích dẫn nguồn web:

Câu hỏi: {question}
"""
        response = model.generate_content(prompt)

        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "Không thể nhận được câu trả lời."

    except Exception as e:
        return f"Lỗi khi gọi API Gemini: {e}"

def web_search(state):
    print("---WEB SEARCH---")
    question = getattr(state, "question")
    original_question = getattr(state, "original_question", question)
    documents = state.documents

    try:
        search_results = tv_search.invoke(question)
        valid_results = [res for res in search_results if res.get("content") and len(res["content"].strip()) > 30]

        if valid_results:
            web_content = "\n\n".join([f"Source: {res['url']}\n{res['content']}" for res in valid_results])
            print(f"✅ Web search lấy được {len(valid_results)} nguồn dữ liệu.")
            web_doc = Document(page_content=web_content)
            documents.append(web_doc)
        else:
            print("⚠️ Không có dữ liệu web phù hợp. Chuyển sang LLM fallback.")
            fallback_answer = call_gemini_free_search(question)
            print(f"✅ LLM Fallback: {fallback_answer}")
            fallback_doc = Document(page_content=f"[LLM Answer Fallback]\n{fallback_answer}")
            documents.append(fallback_doc)

    except Exception as e:
        print(f"❌ Web search error: {e} – fallback sang OpenAI.")
        fallback_answer = call_gemini_free_search(question)
        print(f"✅ LLM Fallback: {fallback_answer}")
        fallback_doc = Document(page_content=f"[LLM Answer Fallback]\n{fallback_answer}")
        documents.append(fallback_doc)

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
        generation = "Tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn."
    elif use_sql == "Yes":
        sql_doc = next((doc for doc in documents if doc.page_content.startswith("SQL used:")), None)
        result_doc = next((doc for doc in documents if doc.page_content.startswith("SQL Query Results:")), None)

        sql_code = ""
        result_table = ""

        if sql_doc:
            sql_code = sql_doc.page_content.replace("SQL used:", "").strip()

        if result_doc:
            result_table = result_doc.page_content.replace("SQL Query Results:", "").strip()

        generation = f"""### Kết quả từ truy vấn cơ sở dữ liệu

#### Câu lệnh SQL được sử dụng:
```sql
{sql_code}
```

#### Kết quả truy vấn:

{result_table}
"""
    else:
        unique_docs = list({doc.page_content: doc for doc in documents}.values())
        formatted_context = format_docs(unique_docs)
        generation = call_gemini_rag(question, formatted_context)

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
    elif web_search_needed == "Yes" and ENABLE_WEB_SEARCH:
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