from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tv_search = TavilySearchResults(max_results=3, search_depth='advanced', max_tokens=10000, api_key=TAVILY_API_KEY)

def web_search(state):
    print("---WEB SEARCH---")
    question = getattr(state, "question")
    original_question = getattr(state, "original_question", question)
    documents = state.documents
    try:
        search_results = tv_search.invoke(question)
        web_content = "\n\n".join([f"Source: {res['url']}\n{res['content']}" for res in search_results])
        web_doc = Document(page_content=web_content)
        documents.append(web_doc)
        print(f"Added web search results ({len(search_results)} sources)")
    except Exception as e:
        print(f"Error during web search: {e}")
    
    return {"documents": documents, "question": original_question}

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = getattr(state, "web_search_needed", "No")
    use_sql = getattr(state, "use_sql", "No")
    
    if use_sql == "Yes":
        print("---DECISION: QUERY SQL DATABASE---")
        return "query_sql"
    elif web_search_needed == "Yes":
        print("---DECISION: WEB SEARCH NEEDED, REWRITE QUERY---")
        return "rewrite_query"
    else:
        print("---DECISION: DOCUMENTS ARE RELEVANT, GENERATE ANSWER---")
        return "generate_answer"