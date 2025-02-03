
# EchoQuill - AI Writing Assistant

EchoQuill is an AI-powered writing assistant that helps you analyze your manuscript and writing influences while maintaining your unique voice. By uploading your manuscript and influence texts (in various formats such as PDF, EPUB, or TXT), you can receive AI-generated suggestions and insights based on your writing style and external inspirations.

## Features

- **Document Processing**: Supports PDF, EPUB, and TXT file formats for your manuscript and influence texts.
- **AI-Powered Writing Analysis**: Ask questions about your writing and receive suggestions related to style, themes, pacing, character arcs, and more.
- **Contextual Influence Integration**: Incorporates relevant sections from your influence texts to enhance the analysis, ensuring suggestions align with your unique voice.
- **Streamlit Interface**: An intuitive web interface to easily upload files and interact with the AI assistant.

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- EbookLib
- BeautifulSoup4
- langchain_nvidia_ai_endpoints
- langchain_community.vectorstores
- langchain.text_splitter
- langchain_core.prompts
- langchain_core.output_parsers
- langchain_core.runnables

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/echoquill.git
   cd echoquill
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your NVIDIA API Key:
   - Go to [NVIDIA AI Endpoints](https://developer.nvidia.com/) and get an API key.
   - Set your API key in the script:
     ```python
     os.environ["NVIDIA_API_KEY"] = "your-nvidia-api-key-here"
     ```

## How to Use

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Your Files**:
   - Upload your manuscript and influence texts in TXT, PDF, or EPUB formats.

3. **Process the Files**:
   - Click the "Process Files" button to extract and process the content from the uploaded files.

4. **Analyze Your Writing**:
   - Select a predefined question or input your custom question to receive an analysis based on the uploaded content.

## Workflow

1. **Document Extraction**: The uploaded files (PDF, EPUB, TXT) are processed and converted into text chunks using a text splitter.
2. **Vectorization**: The text chunks are then vectorized using the NVIDIA Embeddings model and stored in FAISS vector stores for fast retrieval.
3. **Writing Analysis**: The AI assistant generates contextual suggestions by comparing your manuscript against your influences, providing advice on character development, themes, pacing, and writing style.

## Example Usage

- **Custom Question**: "How can I improve the pacing of this chapter?"
- **Predefined Questions**:
  - "How can I develop my character arcs while maintaining my writing style and drawing upon my influences?"
  - "What themes from my influences could I incorporate into this scene?"
  - "Suggest ways to improve the pacing of this chapter."
  - "How does my writing style compare to my influences?"
  - "What are the key emotional beats in my current scene?"

