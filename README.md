# Chatbot Financial LangGraph

## Introduce
LangGraph-based system for financial chatbot tasks using:

- **GraphState**: Tracks the state of the question-answering process
- **Vector Store**: Wikipedia content embedded and stored with ChromaDB
- **AI Models**: OpenAI Embeddings and GPT-4 for various NLP tasks
- **Graph Nodes**: Modular functions for query, response, summarization, etc.
- **Decision Logic**: Routes between different data gathering and reasoning methods
  
![output](https://github.com/user-attachments/assets/feccdc3d-fe40-4287-a022-65385eff58a2)

## Quick Setup

```bash
git clone git@github.com:ngohongthong1832004/chatbot-finacial-langgraph.git
cd chatfinan
python -m venv .venv --clear
.venv/Scripts/activate  # On Windows
source .venv/bin/activate    # On Ubuntu
pip install -r requirements.txt
python run.py
```
Open in browser: [http://localhost:8005/](http://localhost:8005/)

## Project Structure

    api/            API router system  
    data/           Unstructured input data  
    database/       Cloud database connection (Supabase)  
    static/         Frontend web UI  
    services/       Core AI LangGraph processing logic  
    wikipedia_db/   Vector data store (Wikipedia content)  
    run.py          Entry point to run the project  

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4 & Embeddings)
- Internet connection

## Authors

Author: [TruongItt](https://github.com/Truong-itt), [Pine](https://github.com/ngohongthong1832004), [HiTran](https://github.com/HiTranh2504), [Tri](https://github.com/trantrongtri04)

