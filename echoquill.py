import streamlit as st
import os
import tempfile
from pathlib import Path
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

os.environ["NVIDIA_API_KEY"] = "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"

class DocumentProcessor:
    """Handles different document formats and extracts text"""
    
    @staticmethod
    def extract_from_pdf(file):
        """Extract text from PDF file"""
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def extract_from_epub(file):
        """Extract text from EPUB file"""
        book = epub.read_epub(file)
        texts = []
        
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            # Extract text, removing HTML tags
            texts.append(soup.get_text())
            
        return "\n".join(texts)
    
    @staticmethod
    def extract_from_txt(file):
        """Extract text from TXT file"""
        return file.getvalue().decode('utf-8')
    
    @classmethod
    def process_file(cls, file):
        """Process file based on its extension"""
        file_extension = Path(file.name).suffix.lower()
        
        if file_extension == '.pdf':
            return cls.extract_from_pdf(file)
        elif file_extension == '.epub':
            # Save EPUB temporarily as ebooklib requires a file path
            with tempfile.NamedTemporaryFile(suffix='.epub', delete=False) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                return cls.extract_from_epub(tmp_file_path)
            finally:
                os.unlink(tmp_file_path)
        elif file_extension == '.txt':
            return cls.extract_from_txt(file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

class EchoQuillProcessor:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.influences_path = os.path.join(self.temp_dir, 'influences')
        self.manuscript_path = os.path.join(self.temp_dir, 'manuscript')
        os.makedirs(self.influences_path, exist_ok=True)
        os.makedirs(self.manuscript_path, exist_ok=True)
        
        # Initialize models
        self.llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", max_tokens=2048)
        self.embedder = NVIDIAEmbeddings(model="NV-Embed-QA", truncate="END")
        self.ranker = NVIDIARerank(model='nv-rerank-qa-mistral-4b:1', top_n=5)
        
        # Text splitter for processing documents
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        
        # Initialize stores as None
        self.influences_store = None
        self.manuscript_store = None
        
    def process_uploaded_files(self, influence_files, manuscript_file):
        """Process uploaded files and create vector stores"""
        # Process influences
        influence_docs = []
        influence_metadata = []
        
        for file in influence_files:
            try:
                text = DocumentProcessor.process_file(file)
                chunks = self.splitter.split_text(text)
                influence_docs.extend(chunks)
                influence_metadata.extend([{"source": file.name, "type": "influence"}] * len(chunks))
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
        
        # Process manuscript
        try:
            manuscript_text = DocumentProcessor.process_file(manuscript_file)
            manuscript_chunks = self.splitter.split_text(manuscript_text)
            manuscript_docs = manuscript_chunks
            manuscript_metadata = [{"source": manuscript_file.name, "type": "manuscript"}] * len(manuscript_chunks)
        except Exception as e:
            st.error(f"Error processing manuscript: {str(e)}")
            return 0, 0
        
        # Create vector stores
        self.influences_store = FAISS.from_texts(influence_docs, self.embedder, metadatas=influence_metadata)
        self.manuscript_store = FAISS.from_texts(manuscript_docs, self.embedder, metadatas=manuscript_metadata)
        
        return len(influence_docs), len(manuscript_chunks)

    def analyze_writing(self, question):
        """Analyze the writing based on the question"""
        if not self.influences_store or not self.manuscript_store:
            return "Please upload your manuscript and influence files first."
            
        # Create retrievers
        influences_retriever = self.influences_store.as_retriever(search_kwargs={'k': 5})
        manuscript_retriever = self.manuscript_store.as_retriever(search_kwargs={'k': 3})
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are EchoQuill, an AI writing assistant. Analyze the writer's work and their influences to provide helpful suggestions while maintaining their unique voice.
            
            Writer's current work:
            {manuscript_context}
            
            Relevant influences:
            {influences_context}
            """),
            ("user", "{question}")
        ])
        
        # Build chain
        def get_contexts(inputs):
            return {
                "manuscript_context": manuscript_retriever.get_relevant_documents(inputs["question"]),
                "influences_context": influences_retriever.get_relevant_documents(inputs["question"]),
                "question": inputs["question"]
            }
        
        chain = (
            RunnableParallel({"question": RunnablePassthrough()})
            | get_contexts
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(question)

# Streamlit Interface
st.set_page_config(page_title="EchoQuill", page_icon="", layout="wide")

st.title("EchoQuill - AI Writing Assistant")
st.write("Upload your manuscript and influence texts to get AI-powered writing assistance while maintaining your unique voice.")

# Initialize session state for processor
if 'processor' not in st.session_state:
    st.session_state.processor = EchoQuillProcessor()

# File upload section
st.header("Upload Your Writing")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Manuscript")
    manuscript_file = st.file_uploader(
        "Upload your manuscript", 
        type=["txt", "pdf", "epub"], 
        key="manuscript",
        help="Supported formats: TXT, PDF, EPUB"
    )

with col2:
    st.subheader("Your Influences")
    influence_files = st.file_uploader(
        "Upload your influence texts", 
        type=["txt", "pdf", "epub"], 
        accept_multiple_files=True, 
        key="influences",
        help="Supported formats: TXT, PDF, EPUB"
    )

# Process files when uploaded
if manuscript_file and influence_files:
    if st.button("Process Files"):
        with st.spinner("Processing your files..."):
            try:
                influence_chunks, manuscript_chunks = st.session_state.processor.process_uploaded_files(influence_files, manuscript_file)
                st.success(f"Successfully processed {influence_chunks} influence chunks and {manuscript_chunks} manuscript chunks!")
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

# Analysis section
st.header("Writing Analysis")
st.write("Ask questions about your writing or get suggestions for improvement.")

# Predefined questions
question_type = st.selectbox(
    "Choose a question type or write your own below:",
    [
        "Custom question...",
        "How can I develop my character arcs while maintaining my writing style and drawing upon my influences?",
        "What themes from my influences could I incorporate into this scene?",
        "Suggest ways to improve the pacing of this chapter.",
        "How does my writing style compare to my influences?",
        "What are the key emotional beats in my current scene?",
    ]
)

if question_type == "Custom question...":
    question = st.text_input("Enter your question:")
else:
    question = question_type

if question:
    if st.button("Analyze"):
        with st.spinner("Analyzing your writing..."):
            analysis = st.session_state.processor.analyze_writing(question)
            st.write("### Analysis")
            st.write(analysis)