#  DENTSU CREATIVE META ANALYZER

This application is a Streamlit-based tool designed to process, summarize, and analyze content from PDF documents. Utilizing OpenAI's GPT models, it offers functionalities for chunk-based text summarization and question-answering based on the summarized content.

## Features

- **PDF Processing**: Upload and extract text from PDF documents. 
- **Text Summarization**: Summarize large texts by breaking them into manageable chunks.
- **Question Answering**: Answer questions based on the summarized content.
- **Rate Limit Management**: Implement retries with exponential backoff for handling API rate limits.

## Installation

To set up and run the application locally, follow these steps:

1. Clone the repository\
   `git clone guillaumeisobar/DCmeta-analyzer`
2. Navigate to the application directory\
   `cd DCmeta-analyzer`
3. Install the required dependencies\
   `pip install -r requirements.txt`
4. Run the Streamlit application\
   `python3 -m streamlit run app.py   `

## Disclaimer on the Use of pysqlite3

This application uses the `pysqlite3` library as a workaround to address compatibility issues with the SQLite version available on Streamlit Cloud.

`pysqlite3` is used as a drop-in replacement for the standard `sqlite3` library provided by Python. 

This approach allows the application to leverage features available in newer versions of SQLite that may not be present in the version installed on Streamlit Cloud's servers.

## Important Points

### Compatibility

 The use of `pysqlite3` ensures compatibility with certain features and functionalities of SQLite required by the application, which may not be supported by older SQLite versions on Streamlit Cloud.

### Deployment Consideration

When deploying this application on platforms other than Streamlit Cloud, it may be necessary to adjust the implementation or dependencies, especially if the native SQLite version meets the application's requirements.

### Third-Party Library

As `pysqlite3` is a third-party library, developers should review its documentation and source for a full understanding of its functionalities and limitations. This workaround is specific to the deployment environment (Streamlit Cloud) and the requirements of this application at the time of development. Developers should evaluate the necessity of this approach based on their deployment strategy and the evolution of the deployment environments' capabilities.

## Usage

1. Start the Application: Run the application and navigate to the provided URL.
2. Upload PDFs: Use the file uploader to select and upload PDF documents.
3. Process and Summarize: Click the "Process PDFs" button to extract and summarize the text.
4. Ask Questions: Enter your question in the provided text box and click "Ask Question" to get answers based on the summarized content.

## Configuration

OpenAI API Key: Set your OpenAI API key in the \`.env\` file or as an environment variable\
`OPENAI_API_KEY=your_api_key_here`

## Technologies used

- Streamlit
- OpenAI GPT Models
- Python Libraries: pdfplumber, tenacity, dotenv

## Author

Guillaume Olivieri for Dentsu Creative