import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

try:
    # Load HTML file
    with open("../../final-data/readme.htm", "r", encoding="windows-1252") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract tables separately with their context
    tables = []
    for table in soup.find_all('table'):
        # Find the nearest caption or heading
        caption = table.find_previous('p', class_='MsoCaption')
        heading = table.find_previous(['h1', 'h2', 'h3', 'h4'])
        
        context = ""
        if caption:
            context += f"Caption: {caption.get_text().strip()}\n"
        if heading:
            context += f"Section: {heading.get_text().strip()}\n"
        
        # Convert table to text format
        table_text = context + "\nTable Content:\n"
        
        # Get headers
        headers = []
        for th in table.find_all(['th', 'td'], attrs={'style': lambda value: value and 'background:#4F81BD' in value}):
            headers.append(th.get_text().strip())
        
        if headers:
            table_text += " | ".join(headers) + "\n"
            
        # Get rows
        for tr in table.find_all('tr'):
            # Skip header row
            if tr.find(['th', 'td'], attrs={'style': lambda value: value and 'background:#4F81BD' in value}):
                continue
                
            row_cells = []
            for td in tr.find_all(['td', 'th']):
                row_cells.append(td.get_text().strip())
            
            if row_cells:
                table_text += " | ".join(row_cells) + "\n"
        
        tables.append(Document(
            page_content=table_text,
            metadata={"source": "readme.htm", "content_type": "table"}
        ))
    
    # Extract text content from paragraphs
    paragraphs = []
    for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4']):
        # Skip if inside a table
        if p.find_parent('table'):
            continue
            
        text = p.get_text().strip()
        if text and len(text) > 1:  # Only include non-trivial paragraphs
            # Find the nearest heading for context
            heading = p.find_previous(['h1', 'h2', 'h3', 'h4'])
            context = f"Section: {heading.get_text().strip()}\n" if heading else ""
            
            paragraphs.append(Document(
                page_content=context + text,
                metadata={"source": "readme.htm", "content_type": "text"}
            ))
    
    # Combine all documents
    all_docs = tables + paragraphs
    
    # Simple chunking that doesn't require onnxruntime
    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunked_docs = splitter.split_documents(all_docs)
    
    # Create vector store
    chroma_db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=openai_embed_model,
        collection_name="financial_db",
        persist_directory="./financial_db"
    )
    chroma_db.persist()
    print(f"✅ Vector DB đã được tạo với {len(chunked_docs)} chunks.")
    print(f"   - {len(tables)} bảng")
    print(f"   - {len(paragraphs)} đoạn văn bản")

except FileNotFoundError:
    print("❌ Không tìm thấy file HTML tại đường dẫn đã cho.")
except Exception as e:
    print(f"❌ Lỗi xảy ra: {e}")