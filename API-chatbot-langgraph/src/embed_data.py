# scripts/embed_data.py
import os
import gzip
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# Load and preprocess documents
docs = []
with gzip.open('../../data/simplewiki-2020-11-01.jsonl.gz', 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        text = ' '.join(data.get('paragraphs')[:3])
        if 'india' in text.lower():
            docs.append(Document(page_content=text, metadata={"title": data["title"]}))

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
chunked_docs = splitter.split_documents(docs)

# Embed once and persist
chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    embedding=openai_embed_model,
    collection_name="rag_wikipedia_db",
    persist_directory="../wikipedia_db"
)
chroma_db.persist()
print("✅ Vector DB initialized and persisted.")
