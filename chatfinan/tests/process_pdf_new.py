from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import pdfplumber
import re


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_embed_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=OPENAI_API_KEY
)

def extract_income_data(text):
    def find_number(pattern):
        match = re.search(pattern, text)
        return float(match.group(1).replace(',', '')) if match else None
    data = {
        "Net Sales - Products": find_number(r"Products\s+!?\s*([\d,]+)"),
        "Net Sales - Services": find_number(r"Services\s+([\d,]+)"),
        "Total Net Sales": find_number(r"Total net sales.*?([\d,]+)"),
        "Cost of Sales - Products": find_number(r"Cost of sales:\s+Products\s+([\d,]+)"),
        "Cost of Sales - Services": find_number(r"Services\s+([\d,]+)"),
        "Total Cost of Sales": find_number(r"Total cost of sales\s+([\d,]+)"),
        "Gross Margin": find_number(r"Gross margin\s+([\d,]+)"),
        "R&D Expense": find_number(r"Research and development\s+([\d,]+)"),
        "SG&A Expense": find_number(r"Selling, general and administrative\s+([\d,]+)"),
        "Total Operating Expense": find_number(r"Total operating expenses\s+([\d,]+)"),
        "Operating Income": find_number(r"Operating income\s+([\d,]+)"),
        "Other Income": find_number(r"Other income/\(expense\), net\s+\(?-?([\d,]+)"),
        "Income Before Tax": find_number(r"Income before provision for income taxes\s+([\d,]+)"),
        "Income Tax": find_number(r"Provision for income taxes\s+([\d,]+)"),
        "Net Income": find_number(r"Net income\s+!?([\d,]+)"),
        "EPS Basic": find_number(r"Earnings per share:.*?Basic\s+!?([\d\.]+)"),
        "EPS Diluted": find_number(r"Diluted\s+!?([\d\.]+)"),
        "Shares Basic": find_number(r"Shares used in computing earnings per share:\s+Basic\s+([\d,]+)"),
        "Shares Diluted": find_number(r"Diluted\s+([\d,]+)")
    }

    return data

def add_documents_to_vector_db(chroma_db, documents):
    chroma_db.add_documents(documents)
    print(f"✅ Added {len(documents)} documents to the vector DB.")

def process_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    income_data = extract_income_data(text)
    print("\n📊 Báo Cáo Kết Quả Kinh Doanh - Apple (FY24 Q1):\n")
    text = ""
    for k, v in income_data.items():
        print(f"{k}: {v:,.0f}" if isinstance(v, float) else f"{k}: {v}")
        text += f"{k}: {v:,.0f}" if isinstance(v, float) else f"{k}: {v} \n"
    total_Data = "đây là dữ liệu từ file:"+ file_path + "\n" + text 
    documents = [
        Document(page_content=total_Data, metadata={"source": file_path})
    ]
    return documents

def load_vector_store():
    print(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"BASE_DIR: {BASE_DIR}")
    persist_path = os.path.abspath(os.path.join(BASE_DIR, "../wikipedia_db"))

    print(f"📁 Loading vector DB from: {persist_path}")
    files = os.listdir(persist_path)
    print(f"📁 Files in vector DB directory: {files}")

    try:
        chroma_db = Chroma(
            collection_name='rag_wikipedia_db',
            embedding_function=openai_embed_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=persist_path
        )
        doc_count = chroma_db._collection.count()
        if doc_count == 0:
            print("⚠️ No documents found in the vector DB. Adding documents.")
        return chroma_db
    except Exception as e:
        print(f"❌ Failed to load Chroma DB: {e}")
        return None

def print_vector_store_data(chroma_db):
    if chroma_db is not None:
        retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})  
        query = "Báo Cáo Kết Quả Kinh Doanh - Apple (FY24 Q1)"
        documents = retriever.invoke(query)
        
        if documents:
            print("📁 Retrieved documents from the vector DB:")
            for doc in documents:
                print(f"Document Metadata: {doc.metadata}")
                print(f"Document Content: {doc.page_content[:500]}...") 
        else:
            print("⚠️ No relevant documents found.")
    else:
        print("❌ No vector store initialized.")

def process_and_add_documents(file_paths, chroma_db):
    documents = []
    for file_path in file_paths:
        file_path = f"../data/finacial statements/{file_path}.pdf"
        documents += process_pdf(file_path)         
    if documents:
        add_documents_to_vector_db(chroma_db, documents)
def main():
    file_paths = ["FY24_Q1_Consolidated_Financial_Statements", "FY24_Q2_Consolidated_Financial_Statements", 
                  "FY24_Q3_Consolidated_Financial_Statements", "FY24_Q4_Consolidated_Financial_Statements"]
    chroma_db = load_vector_store()
    if chroma_db:
        process_and_add_documents(file_paths, chroma_db)
    chroma_db = load_vector_store()
    if chroma_db:
        print_vector_store_data(chroma_db)
    else:
        print("❌ Failed to load vector store.")
    
if __name__ == "__main__":
    main()
