import os
import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS

# Read Gemini API key from a file
api_key_path = r"C:\Users\gupta\GeminiApikey.txt" 
with open(api_key_path, "r") as file:
    api_key = file.read().strip()

# Set API key in environment
os.environ["GOOGLE_API_KEY"] = api_key

# Streamlit UI
st.title("KhabriBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Get URLs from user input
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
urls = [url for url in urls if url.strip()]  # Remove empty URLs

# Process URLs button
process_url_clicked = st.sidebar.button("Process URLs")

# FAISS index directory
index_dir = "faiss_index"

# Initialize LLM with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9, max_tokens=500)

if process_url_clicked:
    if not urls:
        st.error("❌ Please enter at least one valid URL!")
    else:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        st.text("📡 Fetching data from URLs...✅")
        data = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        st.text("🔄 Splitting text into chunks...✅")
        docs = text_splitter.split_documents(data)

        # Create embeddings with Gemini
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Generate FAISS vector store
        vectorstore_gemini = FAISS.from_documents(docs, embeddings)
        st.text("🛠️ Creating FAISS index...✅")

        # Save FAISS index locally
        vectorstore_gemini.save_local(index_dir)
        st.success("✅ FAISS index saved successfully!")

# Question input for user
query = st.text_input("🔍 Ask a question about the articles:")
if query:
    if os.path.exists(index_dir):
        # Load FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        
        # Create a retrieval chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("💡 Answer")
        st.write(result["answer"])

        # Display sources (if available)
        sources = result.get("sources", "")
        if sources:
            st.subheader("📌 Sources:")
            sources_list = sources.split("\n")  # Split sources by newline
            for source in sources_list:
                st.write(source)
    else:
        st.error("❌ FAISS index not found! Please process URLs first.")
