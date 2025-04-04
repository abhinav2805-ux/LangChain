from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key from file
api_key_path = r"C:\Users\gupta\GeminiApikey.txt"
with open(api_key_path, "r") as file:
    api_key = file.read().strip()
os.environ["GOOGLE_API_KEY"] = api_key

# Load environment variables
load_dotenv()

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.8)

# Initialize instructor embeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

vectordb_file_path = "faiss_index"

def create_vector_db():
    """Loads FAQ data from CSV and creates a FAISS vector database."""
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()
    
    # Create FAISS vector database
    vectordb = FAISS.from_documents(data, instructor_embeddings)
    
    # Save FAISS index
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    """Loads the FAISS vector database and returns a RetrievalQA chain."""
    
    # Ensure FAISS index exists before loading
    if not os.path.exists(vectordb_file_path):
        print("FAISS index not found. Creating new index...")
        create_vector_db()
    
    # Load FAISS vector database
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    
    CONTEXT: {context}
    
    QUESTION: {question}"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})
    return chain
