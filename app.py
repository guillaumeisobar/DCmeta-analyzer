__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from streamlit import logger
# import sqlite3

from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import pdfplumber
from dotenv import load_dotenv
import openai
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables

openai.api_key = os.getenv('OPENAI_API_KEY')

#function to remove duplicate content
def condense_text(text):
    sentences = set()
    condensed = []
    for sentence in text.split('.'):
        trimmed_sentence = sentence.strip()
        if trimmed_sentence and trimmed_sentence not in sentences:
            condensed.append(trimmed_sentence)
            sentences.add(trimmed_sentence)
    return '. '.join(condensed)

def process_pdfs(uploaded_files):
    pdf_texts = []
    for uploaded_file in uploaded_files:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        page_text = condense_text(page_text)  # Call the condense_text function here
                        pdf_texts.append(page_text)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    return "\n".join(pdf_texts)

# Logical chunk text function
def chunk_text(text, chunk_size=500):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += '\n' + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Summarize a single chunk
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
def summarize_chunk(chunk):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt="Summarize the following text:\n\n" + chunk,
            max_tokens=90,
            temperature=0.2
        )
        return response.choices[0].text.strip()
    except openai.error.RateLimitError as e:
        raise

# Function to summarize text with progress bar
def summarize_text(text):
    chunks = chunk_text(text, chunk_size=500)
    total_chunks = len(chunks)
    progress_bar = st.progress(0)
    summaries = []

    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(chunk)
        summaries.append(summary)
        progress_bar.progress((i + 1) / total_chunks)

    progress_bar.empty()
    return "\n".join(summaries)

# Function to handle questions
import tempfile

def handle_question(query, summarized_text):
    # Save the summarized text to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        tmp_file.write(summarized_text)
        tmp_file_path = tmp_file.name

    try:
        # Load the document from the temporary file
        loader = TextLoader(tmp_file_path)
        documents = loader.load()

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(documents, embeddings)
        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.2)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

        response = qa.run(query)
        return response.strip()
    finally:
        # Clean up: remove the temporary file
        os.remove(tmp_file_path)

# Streamlit App
def main():
    st.title("DENTSU CREATIVE META ANALYZER")

    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Process PDFs"):
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                combined_text = process_pdfs(uploaded_files)
                st.session_state.summarized_text = summarize_text(combined_text)
                st.success("PDFs Processed and Summarized")
        else:
            st.warning("Please upload some PDFs.")

    query = st.text_input("Enter your question:")
    
    if st.button("Ask Question"):
        if query and 'summarized_text' in st.session_state:
            with st.spinner("Finding the answer..."):
                answer = handle_question(query, st.session_state.summarized_text)
                st.write(answer)
        else:
            st.warning("Please process PDFs and enter a question.")

if __name__ == "__main__":
    main()
