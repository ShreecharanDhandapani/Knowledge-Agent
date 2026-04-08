import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

DOCS_PATH = "docs"
CHROMA_PATH = "chroma_db"

def load_and_index_documents():
    print("📄 Loading documents...")
    loader = PyPDFDirectoryLoader(DOCS_PATH)
    documents = loader.load()

    print(f"📄 DEBUG: Loaded documents = {len(documents)}")

    if len(documents) == 0:
        raise ValueError("❌ No documents loaded. Check your docs/ folder.")

    print("Sample document preview:")
    print(documents[0].page_content[:300])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"✂️ DEBUG: Number of chunks = {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("❌ No chunks created. Document text may be empty.")

    print("🔢 Embedding documents (this may take a few minutes)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("✅ Documents indexed and saved to ChromaDB")
    return vectorstore


def load_existing_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore


def build_qa_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain


def get_answer(chain, question):
    result = chain.invoke({"question": question})
    answer = result["answer"]
    sources = result.get("source_documents", [])

    source_info = []
    for doc in sources:
        source_info.append({
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200] + "..."
        })

    return answer, source_info