"""
Oracle 23ai RAG Chatbot - Main Streamlit Application
"""

import os
import time
import logging
import subprocess
import streamlit as st
from pathlib import Path

from llama_index.core.llms import ChatMessage

from config_loader import get_config
from database import get_db_manager
from chat_engine import create_chat_engine, llm_chat

logger = logging.getLogger(__name__)


class ChatbotApp:
    """Main Chatbot Application"""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = get_db_manager()
        self.setup_page()
        self.setup_directories()
        self.initialize_session_state()
    
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title=self.config.app.page_title,
            layout=self.config.app.layout,
            initial_sidebar_state="collapsed" if self.config.ui.get('collapsed_sidebar', True) else "expanded"
        )
    
    def setup_directories(self):
        """Create required directories"""
        self.upload_dir = Path(self.config.documents.upload_dir)
        self.processed_dir = Path(self.config.documents.processed_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            "max_tokens": self.config.rag.max_tokens,
            "temperature": self.config.rag.temperature,
            "top_k": self.config.rag.top_k,
            "top_n": self.config.rag.top_n,
            "messages": [],
            "question_count": 0,
            "similarity": self.config.rag.similarity_threshold,
            "select_model": self.config.generation_model.default_model,
            "chat_history": [],
            "enable_rag": True,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def reset_conversation(self):
        """Reset chat conversation"""
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.question_count = 0
        
        if st.session_state.enable_rag and 'chat_engine' in st.session_state:
            st.session_state.chat_engine.reset()
    
    def save_uploaded_file(self, uploaded_file) -> Path:
        """Save uploaded file to upload directory"""
        file_path = self.upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    def process_rag_query(self, question: str):
        """Process query with RAG"""
        try:
            logger.info("Processing RAG query...")
            logger.info(
                f"Parameters: top_k={st.session_state.top_k}, "
                f"max_tokens={st.session_state.max_tokens}, "
                f"temperature={st.session_state.temperature}, "
                f"similarity={st.session_state.similarity}"
            )
            
            time_start = time.time()
            st.session_state.question_count += 1
            
            logger.info(f"Question #{st.session_state.question_count}: {question}")
            
            # Query chat engine
            if self.config.rag.stream:
                response = st.session_state.chat_engine.stream_chat(question)
                return self.handle_stream_response(response)
            else:
                response = st.session_state.chat_engine.chat(question)
                return self.handle_response(response)
                
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return f"An error occurred: {e}"
    
    def handle_response(self, response) -> str:
        """Handle non-streaming response"""
        output = response.response
        source_nodes = response.source_nodes
        
        # Check for valid sources
        has_valid_sources = False
        if source_nodes:
            for node in source_nodes:
                similarity_score = float(node.node.metadata.get("Similarity Score", 0))
                if similarity_score >= st.session_state.similarity:
                    has_valid_sources = True
                    break
        
        # If no valid sources, show friendly message
        if not has_valid_sources:
            output = "I couldn't find relevant information in the uploaded documents. Please try rephrasing your question or upload additional documents."
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(output)
        
        # Add references if enabled
        if self.config.ui.get('show_references', True) and has_valid_sources:
            with st.expander("📚 View Sources"):
                for idx, node in enumerate(source_nodes, 1):
                    similarity = node.node.metadata.get("Similarity Score", 0)
                    if similarity >= st.session_state.similarity:
                        st.markdown(f"**Source {idx}** (Similarity: {similarity:.2%})")
                        st.markdown(f"*File:* {node.node.metadata.get('file_name', 'Unknown')}")
                        st.markdown(f"*Page:* {node.node.metadata.get('page#', 'Unknown')}")
                        st.markdown(f"*Content:* {node.node.text[:200]}...")
                        st.markdown("---")
        
        st.session_state.messages.append({"role": "assistant", "content": output})
        return output
    
    def handle_stream_response(self, response):
        """Handle streaming response"""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in response.response_gen:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        return full_response
    
    def process_llm_query(self, question: str):
        """Process query without RAG (direct LLM)"""
        try:
            with st.spinner("Generating response..."):
                st.session_state.question_count += 1
                logger.info(f"LLM Question #{st.session_state.question_count}: {question}")
                
                output = llm_chat(question, st.session_state['select_model'])
                
                with st.chat_message("assistant"):
                    st.markdown(output)
                
                st.session_state.messages.append({"role": "assistant", "content": output})
                return output
                
        except Exception as e:
            logger.error(f"Error in LLM query: {e}")
            st.error(f"An error occurred: {e}")
    
    def render_sidebar(self):
        """Render sidebar with file upload"""
        if not self.config.ui.get('enable_sidebar', True):
            return
        
        st.sidebar.markdown("### Document Upload")
        st.sidebar.markdown("---")
        
        # File uploader form
        with st.sidebar.form(key="file-uploader-form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=self.config.ui.get('file_uploader', {}).get('accept_multiple', True),
                type=self.config.documents.supported_formats,
                label_visibility="collapsed"
            )
            
            submitted = st.form_submit_button(
                "Upload",
                type="primary",
                use_container_width=True,
                on_click=self.reset_conversation
            )
        
        return submitted, uploaded_files
    
    def process_file_uploads(self, uploaded_files):
        """Process uploaded files"""
        if not uploaded_files:
            return
        
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        existing_docs = self.db_manager.get_existing_documents()
        uploaded_paths = []
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name in existing_docs:
                st.error(
                    f"Document {uploaded_file.name} already exists. "
                    "Please try another document or begin asking questions."
                )
            else:
                file_path = self.save_uploaded_file(uploaded_file)
                uploaded_paths.append(file_path)
                logger.info(f"Uploaded: {uploaded_file.name}")
        
        if uploaded_paths:
            self.run_document_processing(uploaded_files[-1].name)
    
    def run_document_processing(self, last_filename: str):
        """Run document processing script"""
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        try:
            with st.spinner(f"Processing {last_filename}..."):
                logger.info("Starting document processing...")
                
                process = subprocess.Popen(
                    ["python", "process_documents.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                total_steps = 100
                current_step = 0
                
                while process.poll() is None:
                    output = process.stdout.readline()
                    if output:
                        current_step += 1
                        progress = min(current_step / total_steps, 1.0)
                        progress_bar.progress(progress)
                        progress_text.text(output.strip())
                        time.sleep(0.1)
                
                # Read remaining output
                for output in process.stdout:
                    if output:
                        current_step += 1
                        progress = min(current_step / total_steps, 1.0)
                        progress_bar.progress(progress)
                        progress_text.text(output.strip())
                
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    progress_bar.progress(1.0)
                    progress_text.text("✅ Processing complete! Ready to answer questions.")
                else:
                    st.error(f"Processing failed: {stderr}")
                    logger.error(f"Processing failed: {stderr}")
                    
        except subprocess.CalledProcessError as e:
            st.error(f"Error processing document: {e}")
            logger.error(f"Error processing document: {e}")
    
    def display_chat_messages(self):
        """Display chat message history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def run(self):
        """Main application loop"""
        # Page title
        st.markdown(f"<h2 style='text-align: center;'>🔍 {self.config.app.title}</h2>", unsafe_allow_html=True)
        
        # Sidebar
        file_upload_result = self.render_sidebar()
        
        # Clear chat button
        _, col_btn = st.columns([5, 1])
        col_btn.button(
            self.config.ui.get('chat', {}).get('clear_button_text', 'Clear Chat History'),
            type="primary",
            on_click=self.reset_conversation
        )
        
        # Initialize chat engine
        if "messages" not in st.session_state:
            self.reset_conversation()
        
        # Create chat engine only once
        if st.session_state.enable_rag and 'chat_engine' not in st.session_state:
            with st.spinner("Initializing chat engine..."):
                st.session_state.chat_engine, st.session_state.token_counter = create_chat_engine(
                    verbose=False,
                    top_k=st.session_state.top_k,
                    max_tokens=st.session_state.max_tokens,
                    temperature=st.session_state.temperature,
                    top_n=st.session_state.top_n
                )
        
        # Display chat history
        self.display_chat_messages()
        
        # Handle file uploads
        if file_upload_result:
            submitted, uploaded_files = file_upload_result
            if submitted and uploaded_files:
                self.process_file_uploads(uploaded_files)
        
        # Chat input
        placeholder = self.config.ui.get('chat', {}).get('input_placeholder', 'How can I help you?')
        question = st.chat_input(placeholder)
        
        if question:
            # Display user message
            st.chat_message("user").markdown(question)
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.chat_history.append(ChatMessage(role="user", content=question))
            
            # Process based on mode
            if st.session_state.enable_rag:
                self.process_rag_query(question)
            else:
                self.process_llm_query(question)


def main():
    """Main entry point"""
    app = ChatbotApp()
    app.run()


if __name__ == "__main__":
    main()
