import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print(f"Created {docs_path} directory. Please add .txt files and restart.")
        return []
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into smaller chunks"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def get_vector_store(chunks=None, persist_directory="db/chroma_db"):
    """Create or load ChromaDB vector store using local Qwen3 embeddings""" 
    embedding_model = OllamaEmbeddings(model="qwen3-embedding:4b")
    
    if os.path.exists(persist_directory) and chunks is None:
        print("Loading existing vector store...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
    
    print("Creating new vector store...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

def main():
    docs_path = "docs"
    persist_dir = "db/chroma_db"
    
    # 1. Load & Process if DB doesn't exist
    if not os.path.exists(persist_dir):
        documents = load_documents(docs_path)
        if not documents: return
        chunks = split_documents(documents)
        vectorstore = get_vector_store(chunks, persist_dir)
    else:
        vectorstore = get_vector_store(persist_directory=persist_dir)

    # 2. Setup Local LLM (Llama3:8b)
    llm = ChatOllama(model="llama3:8b", temperature=0)

    # 3. Create RAG Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # 4. Interactive Query Loop
    print("\n=== Local RAG System Ready ===")
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() in ['exit', 'quit']: break
        
        response = rag_chain.invoke(query)
        print(f"\nAI: {response['result']}")

if __name__ == "__main__":
    main()