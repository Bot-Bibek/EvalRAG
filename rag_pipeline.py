from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def build_vector_stor(texts: list[str]) -> Chroma:
    """
    Splits raw text into chunks and indexes them in an in-memory ChromaDB store.
    Uses all-MiniLM-L6-v2 embeddings runs locally, no API key needed.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
    )
    docs = [Document(page_content=t) for t in texts]
    split_docs = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return Chroma.from_documents(split_docs, embeddings)

def rag_answer(question: str, vector_store: Chroma) -> tuple[str, list[str]]:
    """
    Runs the full RAG pipline for one question.
    Returns the generated answer AND the retrieved context chunks as plain strings
    RAGAS needs the contexts separately to evaluate faithfulness and context quality.
    """
    
    #Retrieve the top 3 most relevant chunks 
    retriver = vector_store.as_retriever(search_kwargs={"k":3})
    retrieved_docs = retriver.invoke(question)
    
    #RAGAS needs plain strings, not langChain Document objects
    retrieved_contexts = [doc.page_content for doc in retrieved_docs]
    
    #Build a prompt that stuffs the retrieved context in 
    context_block = "\n\n".join(retrieved_contexts)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY
    the information in the context below. If the answer is not in the context,
    say "I don't know."
    
    Context:
    {context_block}
    
    Question: {question}
    
    Answer:
    """
    
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    response = llm.invoke(prompt)
    
    return response.content, retrieved_contexts
    