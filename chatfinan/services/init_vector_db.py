import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# model embedding
openai_embed_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=OPENAI_API_KEY
)
chroma_db = None
similarity_threshold_retriever = None
# create vector store
def init_vector_store():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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