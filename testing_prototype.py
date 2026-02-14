import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate


def load_documents(docs_path="contacts"):
    """
    Recursively loads all prospect essays from the contacts subdirectories.
    """
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print(f"📁 Created {docs_path} directory. Add prospect subfolders and restart.")
        return []
    
    print(f"🔍 Searching for essays in {docs_path}...")
    
    # Use recursive=True to find files in subfolders (contacts/Name/prospect_essay.txt)
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/prospect_essay.txt",  # specifically targets the generated essays
        loader_cls=TextLoader,
        recursive=True 
    )
    
    documents = loader.load()
    
    if not documents:
        print("⚠️ No 'prospect_essay.txt' files found. Ensure you ran the essay generation script first.")
    else:
        print(f"✅ Loaded {len(documents)} prospect essays.")
        
    return documents

def load_document_by_name(person_name, base_path="contacts"):
    """
    Loads the essay for a specific person by their name.
    """
    # 1. Format the name to match the folder naming convention
    folder_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip()
    specific_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(specific_path):
        print(f"❌ Error: No folder found for '{person_name}' at {specific_path}")
        return []

    # 2. Initialize loader for just this subfolder
    loader = DirectoryLoader(
        path=specific_path,
        glob="prospect_essay.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if documents:
        print(f"✅ Loaded document for {person_name}")
    else:
        print(f"⚠️ Found folder for {person_name}, but 'prospect_essay.txt' is missing.")
        
    return documents

def split_documents(documents, chunk_size=100, chunk_overlap=10):
    """Split documents into smaller chunks"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def get_vector_store(chunks=None, persist_directory="prospects_db/chroma_db"):
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

def get_whatsapp_template():
    """Prompt to synthesize info from both databases into a WhatsApp message"""
    template = """
    You are a professional assistant. You have access to two sources of information:
    1. PROSPECT INFO: Details about the person we are contacting.
    2. CAMPAIGN INFO: Details about the offer or marketing message.

    Task: Use the context below to write a personalized WhatsApp message.
    
    Rules:
    - Start with a friendly greeting 👋.
    - Personalize using the Prospect Info.
    - Pitch the offer using the Campaign Info.
    - Use *bold* for key details and bullet points for lists.
    - End with a clear call to action 📅.
    STRICT RULE: Output ONLY the WhatsApp message. Do not include introductory text, 
    conversational filler, or "Here is your message". Start directly with the greeting.

    Context:
    {context}

    Question/Request: {question}
    
    WhatsApp Message:
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_email_template():
    """Prompt for professional Email outreach"""
    template = """
    You are a professional business development representative.
    Context: {context}

    Task: Write a professional outreach email to the prospect.
    
    Rules:
    - Include a clear, catchy **Subject Line**.
    - Use a professional salutation.
    - Mention a specific detail from the Prospect Info to show you've done your research.
    - Connect the Campaign Info to their specific background.
    - Keep it under 150 words.
    - End with a professional sign-off and call to action.

    STRICT RULE: Output ONLY the email content. Do not say "Here is the email" or 
    "Subject: ...". Start immediately with the Subject line and then the body.

    Question/Request: {question}

    Email Format:

    """

    
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_linkedin_template():
    """Prompt for a concise LinkedIn Direct Message"""
    template = """
    You are a networking expert on LinkedIn.
    Context: {context}

    Task: Write a short, engaging LinkedIn DM.
    
    Rules:
    - Keep it brief (LinkedIn users skim!).
    - Mention a commonality or a specific achievement from their Prospect Info.
    - State the purpose of the message clearly using the Campaign Info.
    - No subject line needed.
    - End with a low-pressure question to start a conversation.
    STRICT RULE: Output ONLY the message text. No introductions, no conversational 
    acknowledgments. Just the DM.

    Question/Request: {question}

    LinkedIn DM:
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def main_document():
    docs_path = "prospects"
    persist_dir = "prospects_db/chroma_db"
    
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

    prompt_template = get_whatsapp_template()
    # 3. Create RAG Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template}
    )

    # 4. Interactive Query Loop
    print("\n=== Local RAG System Ready ===")
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() in ['exit', 'quit']: break
        
        response = rag_chain.invoke(query)
        print(f"\nAI: {response['result']}")

def main():
    target_name = input("Enter the person's name to load: ")
    
    # 1. Load Data
    documents = load_document_by_name(target_name)
    if not documents:
        return

    # 2. Vector Store setup
    chunks = split_documents(documents)
    vectorstore = get_vector_store(chunks, persist_directory=f"prospects_db/{target_name.replace(' ', '_')}")

    # 3. Choose Format
    print("\nSelect Output Format:")
    print("1. WhatsApp Message")
    print("2. Professional Email")
    print("3. LinkedIn DM")
    choice = input("Enter choice (1-3): ")

    if choice == "2":
        selected_template = get_email_template()
        msg_type = "Email"
    elif choice == "3":
        selected_template = get_linkedin_template()
        msg_type = "LinkedIn DM"
    else:
        selected_template = get_whatsapp_template()
        msg_type = "WhatsApp Message"

    # 4. Setup LLM and RAG
    llm = ChatOllama(model="llama3:8b", temperature=0.7)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": selected_template}
    )

    # 5. Run query
    print(f"\n🚀 Generating {msg_type} for {target_name}...")
    query = f"Write a {msg_type} for {target_name}"
    response = rag_chain.invoke(query)
    
    print(f"\n--- {msg_type.upper()} ---")
    print(response['result'])


if __name__ == "__main__":
    main()