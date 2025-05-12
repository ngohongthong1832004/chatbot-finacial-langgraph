import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_embed_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=OPENAI_API_KEY
)

def load_vector_store():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    persist_path = os.path.abspath(os.path.join(BASE_DIR, "../financial_data_db"))
    try:
        chroma_db = Chroma(
            collection_name='financial_data_db',
            embedding_function=openai_embed_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=persist_path
        )
        doc_count = chroma_db._collection.count()  
        if doc_count > 0:
            print(f"✅ Loaded existing Vector DB with {doc_count} documents.")
        else:
            print("⚠️ No documents found in the vector DB.")
        return chroma_db
    except Exception as e:
        print(f"❌ Failed to load Chroma DB: {e}")
        return None

def get_vector_store_and_retriever():
    chroma_db = load_vector_store()
    if chroma_db is None:
        print("❌ Failed to load vector store, aborting.")
        return None, None

    doc_count = chroma_db._collection.count()  #
    if doc_count == 0:
        print("⚠️ No documents found in the vector DB. No embedding will be created.")
        return None, None
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
    return chroma_db, similarity_threshold_retriever

def retrieve(state):
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state.question
    documents = []
    _, retriever = get_vector_store_and_retriever()
#     prompt = PromptTemplate.from_template("""
# Trả lời câu hỏi sau dựa trên nội dung tài liệu bên dưới:

# Câu hỏi: {question}

# Tài liệu:
# {context}

# Trả lời ngắn gọn, rõ ràng và chính xác:
# """)
    if retriever:
        try:
            # chain = prompt | llm
            documents = retriever.invoke(question)
            if len(documents) == 0:
                print("⚠️ No relevant documents retrieved.")
            else:
                print(f"📁 Retrieved {documents} documents.")
                print(f"✅ Retrieved {len(documents)} documents.")
        except Exception as e:
            print(f"Error during retrieval: {e}")
    else:
        print("❌ No retriever available.")
    return {"documents": documents, "question": question}
