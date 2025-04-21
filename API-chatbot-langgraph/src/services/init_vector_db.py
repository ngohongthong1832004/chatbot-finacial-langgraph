from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Lấy API key từ biến môi trường
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Tạo mô hình embedding
openai_embed_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=OPENAI_API_KEY
)

# Hàm khởi tạo vector store
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


# Khởi tạo vector DB
chroma_db = init_vector_store()

# Thiết lập retriever
similarity_threshold_retriever = None
if chroma_db:
    try:
        similarity_threshold_retriever = chroma_db.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"k": 3, "score_threshold": 0.3}
        )
        print("✅ Retriever initialized with similarity_score_threshold.")
    except Exception as e:
        print(f"⚠️ Failed to init threshold-based retriever: {e}")
        # Fallback: dùng similarity thường nếu gặp lỗi
        similarity_threshold_retriever = chroma_db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        print("🔁 Fallback to regular similarity retriever.")


if similarity_threshold_retriever:
    query = "what is the capital of India?"
    results = similarity_threshold_retriever.invoke(query)
    print(f"🔍 Retrieved {len(results)} documents:")
    for i, doc in enumerate(results):
        print(f"\n--- Document #{i+1} ---\n{doc.page_content[:300]}")
else:
    print("❌ No retriever initialized.")
