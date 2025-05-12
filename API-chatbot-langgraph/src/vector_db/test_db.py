import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Khởi tạo embedding function
openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

try:
    # Load lại ChromaDB từ thư mục đã lưu
    db = Chroma(
        persist_directory="./financial_db",
        embedding_function=openai_embed_model,
        collection_name="financial_db"
    )

    # Thử truy vấn
    query = "EDGAR document là gì?"
    results = db.similarity_search(query, k=3)

    if results:
        print(f"✅ Tìm thấy {len(results)} kết quả liên quan đến truy vấn: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Kết quả {i} ---")
            print(doc.page_content[:500])  # In 500 ký tự đầu
    else:
        print("⚠️ Không tìm thấy kết quả phù hợp. DB có thể rỗng.")

except Exception as e:
    print(f"❌ Lỗi khi truy vấn ChromaDB: {e}")
