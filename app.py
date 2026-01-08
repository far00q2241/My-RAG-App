# Dependencies to be imported 
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

# Page config
st.set_page_config(page_title="PDF Chatbot", layout="centered")

st.title("📄 PDF Chatbot")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file:
    # Read PDF
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Split text (low memory friendly)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=250
    )
    chunks = splitter.split_text(text)

    st.success("PDF loaded successfully!")

    # User question
    question = st.text_input("Ask a question from the PDF")


    if question:
        # 🔥 SMALL MODEL FOR 4 GB RAM
        llm = Ollama(model="phi")

        # Simple context (only first chunks)
        context = " ".join(chunks[:25])

        prompt = f"""
        You are a helpful assistant.
Use only the given context to answer.
If the answer is not present, say you don't know.".

        Context:
        {context}

        Question:
        {question}
        """

        response = llm(prompt)
        st.write("### Answer")
        st.write(response)
