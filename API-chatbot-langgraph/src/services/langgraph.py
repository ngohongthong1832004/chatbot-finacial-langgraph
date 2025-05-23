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
FORCE_SQL_ONLY = True


chunks, index, embedding_model = None, None, None
def call_openrouter(prompt_obj) -> str:
    if hasattr(prompt_obj, "to_string"):
        prompt = prompt_obj.to_string()
    else:
        prompt = str(prompt_obj) 

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
# doc_grader = grade_prompt | llm | StrOutputParser()
doc_grader = grade_prompt | RunnableLambda(call_openrouter) | StrOutputParser()


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
# question_rewriter = (re_write_prompt|llm|StrOutputParser())
question_rewriter = (re_write_prompt|RunnableLambda(call_openrouter)|StrOutputParser())


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
    print(f"📥 Question: {question}")
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

# def is_sql_question(question: str) -> bool:
#     response = llm.invoke([
#         HumanMessage(content=f"""Bạn là chuyên gia phân loại truy vấn. 
# Hãy xác định xem câu hỏi sau đây có yêu cầu truy vấn dữ liệu từ cơ sở dữ liệu SQL hay không. 
# Trả lời chỉ 'yes' hoặc 'no'.

# Câu hỏi: {question}""")
#     ])
#     return response.content.strip().lower() == "yes"

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "Bạn là chuyên gia phân loại truy vấn. Hãy xác định xem câu hỏi sau đây có yêu cầu truy vấn dữ liệu từ cơ sở dữ liệu SQL hay không. Trả lời chỉ 'yes' hoặc 'no'."),
#     ("human", "Câu hỏi: {question}")
# ])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
Bạn là chuyên gia phân loại truy vấn cho hệ thống hỏi đáp tài chính.

Hệ thống có khả năng truy vấn dữ liệu SQL từ cơ sở dữ liệu chứa giá cổ phiếu lịch sử (daily OHLCV) của 30 công ty thuộc chỉ số Dow Jones Industrial Average (DJIA), bao gồm:

- Ticker mã có trong hệ thống 
            AAPL - Apple Inc.
            AMGN - Amgen Inc.
            AXP  - American Express
            BA   - Boeing Co.
            CAT  - Caterpillar Inc.
            CRM  - Salesforce Inc.
            CSCO - Cisco Systems
            CVX  - Chevron Corp.
            DIS  - Walt Disney Co.
            DOW  - Dow Inc.
            GS   - Goldman Sachs
            HD   - Home Depot
            HON  - Honeywell International
            IBM  - International Business Machines
            INTC - Intel Corp.
            JNJ  - Johnson & Johnson
            JPM  - JPMorgan Chase
            KO   - Coca-Cola Co.
            MCD  - McDonald's Corp.
            MMM  - 3M Company
            MRK  - Merck & Co.
            MSFT - Microsoft Corp.
            NKE  - Nike Inc.
            PG   - Procter & Gamble
            TRV  - Travelers Companies
            UNH  - UnitedHealth Group
            V    - Visa Inc.
            VZ   - Verizon Communications
            WBA  - Walgreens Boots Alliance
            WMT  - Walmart Inc.
- Cột dữ liệu: "Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"
- Khoảng thời gian dữ liệu sẵn có:
    + Từ ngày "2023-04-26" đến ngày "2025-04-25" (định dạng YYYY-MM-DD). Nếu câu hỏi yêu cầu dữ liệu nằm **ngoài** khoảng thời gian này, hãy trả lời **no**.

Nhiệm vụ của bạn: xác định xem câu hỏi dưới đây có thể được trả lời bằng dữ liệu trong cơ sở dữ liệu SQL hay không lưu ý nếu thời gian không thuộc vào hệ thống thì là no.
Ngoài ra nếu câu hỏi có thời gian bạn cần phải xác định chính xác thời gian của câu hỏi có nằm trong khoảng thời gian của hệ thống hay không từ format YYYY-MM-DD (quan trọng).
Chỉ trả lời **yes** hoặc **no**, không thêm giải thích.

Nếu câu hỏi đề cập đến giá, thời gian cụ thể, ticker/công ty có trong danh sách, hãy trả lời "yes".
"""),
    ("human", "Câu hỏi: {question}")
])

def is_sql_question(question: str) -> bool:
    # filled_prompt = prompt_template.format(question=question)
    print(f"📥 Question: {question}")
    response = call_openrouter(prompt_template.format_prompt(question=question))
    return response.strip().lower() == "yes"
    # return "yes"


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state.question
    documents = state.documents

    filtered_docs = []
    web_search_needed = "No"
    use_sql = "No"

    # Check SQL query keywords
    # database_keywords = [
    #     "lấy dữ liệu", "database", "sql", "select", "from table", "stocks", "query", "table", 
    #     "dữ liệu", "cơ sở dữ liệu", "truy vấn", "truy vấn sql", "lấy dữ liệu", "sinh sql", "câu lệnh"
    # ]

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
            # print(f"\n📝 Grading document #{i+1}")
            # print(f"📌 Question:\n{question}")
            # print(f"📄 Document Content (preview):\n{d.page_content[:1000]}")
            # print("-" * 60)

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



def generate_chart_code_via_llm(question: str, df: pd.DataFrame) -> str:
    df_sample = df.head(10).to_csv(index=False)
    prompt = f"""
You are a Python expert specialized in data visualization using matplotlib.

The variable `df` is a preloaded pandas DataFrame that contains the dataset. Here are the first 10 rows:

{df_sample}

User question: "{question}"

Instructions:
- Always start your code with `plt.figure(figsize=(10, 7))` and end with `plt.tight_layout()` and `plt.savefig(savefig_path)` and `plt.close()`.
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

        # Chặn các lệnh nguy hiểm
        # forbidden_keywords = ["import", "input(", "os.", "__", "eval(", "exec(", "subprocess"]
        # if any(k in cleaned_code for k in forbidden_keywords):
        #     raise ValueError("⚠️ Unsafe code detected in chart code.")

        # Lọc các dòng hợp lệ bắt đầu bằng df., plt., sns. (bỏ savefig và close)
        filtered_lines = []
        for line in cleaned_code.splitlines():
            line = line.strip()
            if (line.startswith("df") or line.startswith("plt.") or line.startswith("sns.")) and not any(x in line for x in ["savefig", "close"]):
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
    import json
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

#         generation = f"""### Kết quả từ truy vấn cơ sở dữ liệu

# #### Câu lệnh SQL được sử dụng:
# ```sql
# {sql_code}
# ```

# #### Kết quả truy vấn:

# {result_table}

# #### Kết luận:
# """

        generation = f"""Kết luận:"""




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
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df.columns = [c.strip() for c in df.columns]  # xóa khoảng trắng
            df = df.reset_index(drop=True)
            
            # lưu thành file tạm thời
            # temp_file_path = f"temp_{uuid.uuid4().hex}.csv"
            # df.to_csv(temp_file_path, index=False)

            for col in df.select_dtypes(include=['float']).columns:
                if df[col].max() > 1e11:
                    df[col] = df[col] / 1e9  # Convert sang đơn vị tỷ
            # Sinh mã vẽ và render
            chart_code = generate_chart_code_via_llm(question, df)
            chart_url = execute_generated_plot_code(chart_code, df)

            conclusion = generate_sql_conclusion(question, df)
            generation += f"\n{conclusion}"    
            
            if chart_url:
                generation += f"\n\n ![Xem biểu đồ tại đây](http://localhost:8000{chart_url})"
            
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

    if FORCE_SQL_ONLY:
        # Chế độ chỉ test SQL branch
        agentic_rag.add_node("query_sql", query_sql)
        agentic_rag.add_node("generate_answer", generate_answer)
        agentic_rag.set_entry_point("query_sql")
        agentic_rag.add_edge("query_sql", "generate_answer")
        agentic_rag.add_edge("generate_answer", END)
        return agentic_rag.compile()

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