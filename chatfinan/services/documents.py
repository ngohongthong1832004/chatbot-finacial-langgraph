from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from operator import itemgetter
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert grader assessing relevance of a retrieved document to a user question.\nAnswer only 'yes' or 'no' depending on whether the document is relevant to the question."),
    ("human", "Retrieved document:\n{document}\n\nUser question:\n{question}")
])
doc_grader = grade_prompt | llm | StrOutputParser()

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state.question
    documents = state.documents
    
    filtered_docs = []
    web_search_needed = "No"
    use_sql = "No"
    
    # Define database related keywords
    database_keywords = [
        "database", "sql", "select", "from table", "query", "dữ liệu", "cơ sở dữ liệu",
        "bảng", "bản ghi", "giá cổ phiếu", "symbol", "ticker", "ticker symbol",
        "giá mở cửa", "giá đóng cửa", "tăng trưởng", "lợi tức", "dividend",
        "PE ratio", "market cap", "volume", "doanh nghiệp", "công ty", "chỉ số DJIA",
        "truy vấn", "thống kê", "so sánh", "câu lệnh SQL", "tham chiếu", "foreign key",
        "bảng giá", "bảng công ty", "liên kết bảng", "tham chiếu bảng", "trường dữ liệu"
    ]

    # Check if question is related to database first
    if any(keyword in question.lower() for keyword in database_keywords):
        use_sql = "Yes"
        print("---GRADE: DETECTED DATABASE QUERY---")
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search_needed": "No",
            "use_sql": "Yes"
        }
    
    if documents:
        for d in documents:
            score = doc_grader.invoke({
                "question": question,
                "document": d.page_content
            })
            if score.strip().lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT, SUGGEST WEB SEARCH---")
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

re_write_prompt = ChatPromptTemplate.from_template(
    """Given a user question, rewrite it to be a more effective web search query.
Original question: {question}
Search query:"""
)
question_rewriter = (re_write_prompt|llm|StrOutputParser())

def rewrite_query(state):
    print("---REWRITING QUERY FOR WEB SEARCH---")
    question = state.question
    documents = state.documents
    rewritten_question = question_rewriter.invoke({"question": question})
    print(f"Rewritten question: {rewritten_question}")
    return {"documents": documents, "question": rewritten_question, "original_question": question}

def format_docs(docs):
    return "\n\n".join(doc.page_content if isinstance(doc, Document) else str(doc) for doc in docs)

prompt_template = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If no context is present or if you don't know the answer, just say that you don't know the answer.
    Do not make up the answer unless it is there in the provided context.
    Give a detailed answer and to the point answer with regard to the question.
    Question:
    {question}
    Context:
    {context}
    Answer:
    """
)
chatgpt = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)

qa_rag_chain = (
    {
        "context": (itemgetter('context') | RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
    | prompt_template 
    | chatgpt
    | StrOutputParser()
)

def generate_answer(state):
    print("---GENERATE ANSWER---")
    question = state.question
    documents = state.documents
    use_sql = state.use_sql

    if not documents:
        generation = "I don't have enough information to answer this question."
    elif use_sql == "Yes":
        # Tách riêng document chứa SQL và document chứa kết quả
        sql_doc = next((doc for doc in documents if doc.page_content.startswith("SQL used:")), None)
        result_doc = next((doc for doc in documents if doc.page_content.startswith("SQL Query Results:")), None)

        generation_parts = []
        if sql_doc:
            generation_parts.append(sql_doc.page_content)
        if result_doc:
            generation_parts.append(result_doc.page_content)

        generation = "\n\n".join(generation_parts) if generation_parts else "No SQL content found."
    else:
        # Loại bỏ documents trùng lặp
        unique_docs = list({doc.page_content: doc for doc in documents}.values())
        formatted_context = format_docs(unique_docs)

        # print("---FORMATTED CONTEXT PREVIEW---")
        # print(formatted_context[:300])

        generation = qa_rag_chain.invoke({
            "context": formatted_context,
            "question": question
        })

    # print(f"Question: {question}")
    # print(f"documents: {formatted_context}")
    # print(f"Answer: {generation}")
    # print(f"Documents count: {len(documents)}")

    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }