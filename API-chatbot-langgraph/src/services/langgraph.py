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
from deep_translator import GoogleTranslator
from langchain_core.messages import BaseMessage




# from vertexai.generative_models import GenerativeModel

import os
import io  # đừng quên import ở đầu file nếu chưa có
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
import seaborn as sns
        
import os
import matplotlib.pyplot as plt
import pandas as pd
import uuid


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
    
def call_openrouter_for_rewriting(prompt_obj) -> str:
    prompt = prompt_obj.to_string()
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": """ You are a smart assistant who rewrites user questions to make them clearer and more suitable for Google search.
                                Your job is to improve the question for search engines without changing its original meaning or intent.
                                Only make edits that:
                                - clarify vague phrasing,
                                - add keywords for relevance,
                                - or remove unnecessary words.
                                Do NOT change the meaning, time range, company names, or data types.
                                If the question already contains a specific year (e.g., 2024), keep it. Only use 2025 if it is clearly implied or the original question is ambiguous.
                                Output only the rewritten question.
                            """
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    return res.json()["choices"][0]["message"]["content"].strip()

def call_openrouter_for_generic(prompt: str, system_message: str = "You are a helpful assistant.") -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Error from OpenRouter: {e}")
        return "Không thể tạo kết luận từ dữ liệu."



def call_openrouter_for_sql_classification(prompt_obj) -> str:
    prompt = prompt_obj.to_string()  

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert classifier for SQL queries question."},
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
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY) if ENABLE_GPT_GRADING else None
tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000, api_key=TAVILY_API_KEY)


# get_vector_store_and_retriever
def get_vector_store_and_retriever(resource_dir: str = "sec_embeddings") -> Tuple[List[Dict[str, Any]], faiss.Index, SentenceTransformer]:
    global chunks, index, embedding_model
    
    if chunks is not None and index is not None and embedding_model is not None:
        print("Vector store & retriever already initialized. Reusing...")
        return chunks, index, embedding_model

    try:
        # print(f"📁 Loading RAG vector store from: {resource_dir}")
        print(f"📁 Loading RAG vector store")
        
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
# doc_grader = grade_prompt | llm | StrOutputParser()
doc_grader = grade_prompt | RunnableLambda(call_openrouter) | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content if isinstance(doc, Document) else str(doc) for doc in docs)


def call_gemini_rag(question: str, context: str) -> str:
    prompt = f"""
    You are a knowledgeable and reliable AI assistant. Your task is to answer the user's question based solely on the information provided in the context below.

    Please ensure your response:
    - Uses only facts found in the given context.
    - Clearly states if the answer cannot be determined from the context.
    - Does not make up or assume any information not explicitly mentioned.

    Question: {question}

    Context:
    {context}

    Answer:
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()




# Query rewriter for web search
re_write_prompt = ChatPromptTemplate.from_template(
    """Please rewrite the following question to make it concise, clear, and well-suited for a Google search.

Requirements:
- The rewritten question should focus on retrieving the most current and up-to-date information (e.g., during the year 2025).
- Avoid including outdated years or references to obsolete data.

Original question: {question}
Rewritten web search query:"""
)

# question_rewriter = (re_write_prompt|llm|StrOutputParser())
question_rewriter = (re_write_prompt|RunnableLambda(call_openrouter_for_rewriting)|StrOutputParser())


def translate_text(text, to_lang='en'):
    try:
        translated = GoogleTranslator(source='auto', target=to_lang).translate(text)
        return translated
    except Exception as e:
        print(f"❌ Lỗi khi dịch văn bản: {e}")
        return text 

def retrieve(state):
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state.question
    documents = []
    # print(f"📥 Question: {question}")
    question = translate_text(question, to_lang='en')
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



prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert query classifier for a financial Q&A system.

The system has SQL query capabilities for a database containing:

**AVAILABLE DATA:**
- Daily historical stock prices (OHLCV) for 30 companies in the Dow Jones Industrial Average (DJIA)
- Companies: AAPL (Apple), AMGN (Amgen), AXP (American Express), BA (Boeing), CAT (Caterpillar), CRM (Salesforce), CSCO (Cisco), CVX (Chevron), DIS (Disney/Walt Disney), DOW (Dow Inc.), GS (Goldman Sachs), HD (Home Depot), HON (Honeywell), IBM, INTC (Intel), JNJ (Johnson & Johnson), JPM (JPMorgan Chase), KO (Coca-Cola), MCD (McDonald's), MMM (3M), MRK (Merck), MSFT (Microsoft), NKE (Nike), PG (Procter & Gamble), TRV (Travelers), UNH (UnitedHealth Group), V (Visa), VZ (Verizon), WBA (Walgreens Boots Alliance), WMT (Walmart)
- Data columns: "Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "price", "companies", "ticker", "sector", "industry"
- Time range: e.g from "2023-04-26" to "2025-04-25"

**SUPPORTED QUERY TYPES:**

**ALWAYS answer "yes" for questions about:**
- have price data for any of the 30 DJIA companies
- Price data for a specific date or time range
- Price data for a specific company
- Stock prices (closing price, opening price, highest price, lowest price)
- Trading volume
- Dividends and stock splits
- Price comparisons between companies
- Statistical calculations (average, total, percentage change, standard deviation, correlation)
- Trend analysis and performance
- Company rankings by financial criteria
- Finding highest/lowest values within time periods
- Counting trading days meeting specific conditions
- Sector/industry classification
- Ticker symbol information
- Any question related to financial data of the 30 DJIA companies

**TIME PERIOD SUPPORT:**
- Accept any date from 2023-04-26 to 2025-04-25
- Accept time ranges (quarters, years, months) within the above scope
- Examples: "2024", "Q1 2025", "March 2024", "Jan-Mar 2025" are all acceptable

**COMPANY NAMES:**
- Accept both full names and abbreviations
- Examples: "Apple", "Microsoft", "Walt Disney", "Johnson & Johnson", "Procter & Gamble"

**ONLY answer "no" when:**
- Question is about companies NOT in the 30 DJIA company list
- Question is completely unrelated to finance/stocks

**IMPORTANT NOTES:**
- When in doubt, prefer answering "yes" 
- All types of financial analysis, statistics, and comparisons are supported
- Chart/visualization questions are also supported as they can query SQL data for creation
- NEVER answer "no" if the question contains any of these keywords:"Create", "Plot", "price", "prices", "market", "volume", "dividends", "splits", "stock price", "closing price", "opening price", "companies".
- IMPORTANT: If the question have any SQL query keywords and some keyword such as:"Create", "Plot", "price", "market", "volume", "dividends", "splits", "companies" always answer "yes".

### ADDITIONAL RULES FOR CLASSIFICATION:

Evaluate the question using the following logic:

- ✅ Answer "yes" if the question:
  - Involves retrieving or calculating data such as stock prices, volume, dividends, volatility, returns, averages, rankings, or performance
  - Mentions any of these patterns: "compare", "plot", "highest", "lowest", "show", "how much", "calculate", "average", "total", "volume of", "change in price", "between dates", "top companies", "most volatile", "return of"

- ❌ Answer "no" if:
  - The question is theoretical, descriptive, or conceptual (e.g., "What is a dividend?", "Explain stock market volatility")
  - It only asks about general knowledge, not specific data from the database

### EXAMPLES TO GUIDE YOU:

**Answer "yes" if the question is like:**
- "Which stock had the highest closing price in 2024?"
- "Compare the performance of AAPL and MSFT"
- "Show volume trends for Boeing in Q2 2023"
- "List companies with average dividend yield above 2%"
- "What was Apple's max drawdown in 2024?"

**Answer "no" if the question is like:**
- "What is the definition of P/E ratio?"
- "How does the stock market work?"
- "Tell me about Microsoft's history"
- "What does DJIA represent?"

Respond with only one word: **yes** or **no**.
"""),
    ("human", "Question: {question}")
])

def is_sql_question(question: str) -> bool:
    # print(f"📥 Question: {question}")
    response = call_openrouter_for_sql_classification(prompt_template.format_prompt(question=question))
    return response.strip().lower() == "yes"


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state.question
    documents = state.documents

    filtered_docs = []
    web_search_needed = "No"
    use_sql = "No"

    # Grade retrieved documents
    if documents:
        for i, d in enumerate(documents):
            score = doc_grader.invoke({
                "question": question,
                "document": d.page_content,
                "metadata": json.dumps(d.metadata, indent=2)
            })
            if score.strip().lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
        if filtered_docs:
            print(f"---GRADE: {len(filtered_docs)} RELEVANT DOCUMENTS RETRIEVED---")
        else:
            if is_sql_question(question):
                print("---GRADE: DETECTED DATABASE QUERY WITH NO RELEVANCE---")
                use_sql = "Yes"
            else:
                print("---NO DOCUMENTS RELEVANCE RETRIEVED---")
                web_search_needed = "Yes"
    else:
        if is_sql_question(question):
            print("---GRADE: DETECTED DATABASE QUERY---")
            use_sql = "Yes"
        else:
            print("---NO DOCUMENTS RELEVANCE RETRIEVED---")
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



def generate_sql_conclusion(question: str, df: pd.DataFrame) -> str:
    sample = df.head(10).to_dict(orient='records')
    prompt = f"""
Bạn là chuyên gia phân tích dữ liệu. Dưới đây là câu hỏi của người dùng và dữ liệu SQL vừa truy vấn được (gồm 10 dòng đầu tiên).

Câu hỏi: {question}

Dữ liệu:
{json.dumps(sample, ensure_ascii=False, indent=2)}

Hãy viết một đoạn kết luận ngắn gọn (1-2câu) để tóm tắt hoặc đưa ra insight từ dữ liệu này. Không cần giải thích lại câu hỏi. Ngắn ngọn thôi.
"""
    return call_openrouter_for_generic(prompt.strip())



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

        # Generate SQL query
        sql_query = generate_sql_query(question)
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
            # print("📊 Query Results:\n", content)

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
        

def call_openrouter_for_chart(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a Python data visualization expert. Only return matplotlib code using the df variable. Do not explain."},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    return res.json()["choices"][0]["message"]["content"].strip()



def generate_chart_code_via_llm(question: str, df: pd.DataFrame) -> str:
    df_sample = df.head(10).to_csv(index=False)
    prompt = f"""
You are a Python expert specialized in data visualization using matplotlib.

The variable `df` is a preloaded pandas DataFrame that contains the dataset. Here are the first 10 rows:

{df_sample}

User question: "{question}"

Instructions:
- Always start your code with `plt.figure(figsize=(10, 7))` and end with `plt.tight_layout()`.
- Only use the existing `df` variable. Do NOT define or assign any new variables like `df_sorted`, `correlation_matrix`, or `data`.
- If transformation is needed (e.g., sorting, pivoting, grouping), reassign it directly to `df`, like: df = df.sort_values(...).
- You MAY use df['column'].values or df['column'].tolist() inside plotting functions (e.g., for labels and values in pie charts).
- All plotting operations must reference `df` directly.
- Do NOT use `plt.show()`, `import`, `input`, `eval`, `exec`, or any OS/system functions.
- Do NOT use control flow statements such as `if`, `while`, or `for`.
- Every line of code must begin with `df.`, `plt.`, or `sns.` (for seaborn).
- Do NOT include any explanation, comment, markdown formatting, or code fences (```).
- Do NOT include any import statements. All required libraries (pandas, matplotlib, seaborn) are already available.
- Your output will be automatically executed. Only return clean, complete matplotlib code using `df`.
- ALWAYS use keyword arguments in df.pivot(): e.g., df = df.pivot(index="...", columns="...", values="...") — never pass positional arguments.
- If using .dt accessors (e.g., df["Date"].dt.month), make sure to convert "Date" to datetime first using: df["Date"] = pd.to_datetime(df["Date"])



Output only the raw code.
"""

    code = call_openrouter_for_chart(prompt.strip())
    print("📤 Code gen chart from LLM:\n", code)
    return code.strip()



def execute_generated_plot_code(code: str, df: pd.DataFrame, static_dir="static/charts") -> str:
    import os
    os.makedirs(static_dir, exist_ok=True)

    filename = f"chart_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(static_dir, filename)

    local_vars = {
        "df": df.copy(),
        "plt": plt,
        "pd": pd,
        "savefig_path": filepath,
    }

    try:
        # Xoá markdown và các lệnh không an toàn
        cleaned_code = (
            code.replace("```python", "")
                .replace("```", "")
                .replace("plt.show()", "")
                .strip()
        )


        # Lọc các dòng hợp lệ bắt đầu bằng df., plt., sns. (bỏ savefig và close)
        filtered_lines = []
        for line in cleaned_code.splitlines():
            line = line.strip()
            if (line.startswith("df") or line.startswith("plt.") or line.startswith("sns.")):
                if "savefig" not in line and "plt.close" not in line:
                    filtered_lines.append(line)

        if not filtered_lines:
            raise ValueError("⚠️ No valid plotting commands detected.")

        # Ghép code và đảm bảo lưu đúng ảnh
        safe_code = "\n".join(filtered_lines) + "\nplt.savefig(savefig_path)\nplt.close()"


        print("📋 Running cleaned matplotlib code:\n", safe_code)

        exec(safe_code, {"plt": plt, "pd": pd, "sns": sns}, local_vars)

        return f"/static/charts/{filename}"

    except Exception as e:
        print(f"❌ Error in generated chart code: {e}")
        return ""




def generate_sql_conclusion(question: str, df: pd.DataFrame) -> str:
    sample = df.head(10).to_dict(orient='records')
    prompt = f"""
You are a financial data analysis expert.

Question:
{question}

The following data is the result of a SQL query, showing up to 20 rows.
**The first row represents the most significant value (e.g., the highest or lowest depending on the query).**

Data:
{json.dumps(sample, ensure_ascii=False, indent=2)}

Instructions:
- If the question asks for ranking or top-N (e.g., "top 3", "rank", "highest 5", etc.), list the top relevant entries (e.g., top 3 companies).
- Otherwise, write a short and concise conclusion in **one sentence only**, focusing on the most significant result (usually the first row).
- Do NOT restate the question.
- Do NOT provide explanations or background.
- Be direct and clear.

Conclusion:
"""
    return call_openrouter(prompt.strip())



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

#### Kết luận:
"""

        # generation = f"""Kết luận:"""
        try:
            
            if(not result_table):
                generation += "Không có kết quả nào từ truy vấn SQL."
                return {
                    "documents": documents,
                    "question": question,
                    "generation": generation
                }
            
            # Parse markdown table from SQL result
            lines = result_table.strip().splitlines()
            clean_lines = [line for line in lines if "---" not in line]
            table_str = "\n".join(clean_lines)

            # Đọc lại bằng pandas
            df = pd.read_table(io.StringIO(table_str), sep="|", engine='python')
            df = df.dropna(axis=1, how='all')  # bỏ cột rỗng do padding '|'

            # ✅ Không còn dùng applymap
            for col in df.select_dtypes(include='object'):
                df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

            df.columns = [c.strip() for c in df.columns]  # xóa khoảng trắng
            df = df.reset_index(drop=True)
            
            # lưu thành file tạm thời
            # temp_file_path = f"temp_{uuid.uuid4().hex}.csv"
            # df.to_csv(temp_file_path, index=False)

            # for col in df.select_dtypes(include=['float']).columns:
            #     if df[col].max() > 1e11:
            #         df[col] = df[col] / 1e9  # Convert sang đơn vị tỷ
                    
                    
            print(f"📊 DataFrame shape: {df.head(3)}")
                    
            # Sinh mã vẽ và render
            chart_url = None
            if df.shape[1] >= 1:
                try:
                    chart_code = generate_chart_code_via_llm(question, df)
                    chart_url = execute_generated_plot_code(chart_code, df)
                except Exception as e:
                    print(f"⚠️ Không thể tạo biểu đồ: {e}")

            # conclusion = generate_sql_conclusion(question, df)
            # generation += f"\n{conclusion}"    
            
            if chart_url:
                generation += f"\n\n ![Xem biểu đồ tại đây](http://localhost:8000{chart_url})"
            
            
            # lưu thành file tạm thời
            temp_file_path = f"temp.csv"
            df.to_csv(temp_file_path, index=False)
            print(f"✅ Tạo file tạm thời: {temp_file_path}")
            
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ: {e}")
    else:
        unique_docs = list({doc.page_content: doc for doc in documents}.values())
        formatted_context = format_docs(unique_docs)
        generation = call_gemini_rag(question, formatted_context)
    # print(f"📄 Formatted context: {formatted_context[:1000]}...")
    print(f"✅ Generated answer: {generation}...")
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