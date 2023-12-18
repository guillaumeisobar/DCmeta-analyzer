import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import openai
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to process PDFs
def process_pdfs(uploaded_files):
    pdf_texts = []
    for uploaded_file in uploaded_files:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_texts.append(page_text)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    return "\n".join(pdf_texts)

# Chunk text function
def chunk_text(text, chunk_size=1500):
    chunks = []
    current_chunk = ""
    for paragraph in text.split('\n'):
        if len(current_chunk) + len(paragraph) > chunk_size or '\n' in paragraph:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk += '\n' + paragraph
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Summarize a single chunk
def summarize_chunk(chunk):
    response = openai.Completion.create(
        model="text-curie-001",
        prompt="Summarize the following text:\n\n" + chunk,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Function to summarize text with progress bar
def summarize_text(text):
    chunks = chunk_text(text, chunk_size=1500)
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
    st.title("PDF Summarizer and Q&A Tool")

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
