import streamlit as st
from langchain_helper import get_qa_chain

st.title("Course QnA Chatbot")

question = st.text_input("Enter your question:")

if question:
    chain = get_qa_chain()
    response = chain(question)
    
    st.header("Answer")
    st.write(response["result"])
