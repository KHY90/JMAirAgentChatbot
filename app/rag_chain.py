
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from app.supabase_client import supabase

def get_rag_chain():
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize Supabase vector store
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )

    # Initialize retriever
    retriever = vector_store.as_retriever()

    # Define the prompt template
    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the chat model
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
