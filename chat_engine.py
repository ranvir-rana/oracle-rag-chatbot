"""
Chat Engine Module
Creates and manages RAG chat engine with LLM and vector store
"""
import os
import logging
import streamlit as st
from tokenizers import Tokenizer

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.oci_genai import OCIGenAI
from llama_index.embeddings.oci_genai import OCIGenAIEmbeddings
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.llms import ChatMessage

from config_loader import get_config
from oracle_vectorstore import OracleVectorStore
from oci_utils import load_oci_config, print_configuration

logger = logging.getLogger(__name__)


class ChatEngineManager:
    """Manages chat engine creation and configuration"""
    
    def __init__(self):
        self.config = get_config()
        self.oci_config = load_oci_config()
    
    def create_embedding_model(self) -> OCIGenAIEmbeddings:
        """Create embedding model"""
        embed_config = self.config.embedding_model
        
        embed_model = OCIGenAIEmbeddings(
            auth_profile=self.config.oci.profile_name,
            compartment_id=self.config.oci.compartment_ocid,
            model_name=embed_config.model_name,
            truncate=embed_config.truncate,
            service_endpoint=self.config.oci.endpoint,
        )
        
        logger.info(f"Created embedding model: {embed_config.model_name}")
        return embed_model
    
    def create_llm(self, model_name: str = None) -> OCIGenAI:
        """Create LLM instance"""
        gen_config = self.config.generation_model
        
        if model_name is None:
            # Get from session state or use default
            if hasattr(st, 'session_state') and 'select_model' in st.session_state:
                model_name = st.session_state['select_model']
            else:
                model_name = gen_config.default_model
        
        llm = OCIGenAI(
            model=model_name,
            service_endpoint=self.config.oci.endpoint,
            compartment_id=self.config.oci.compartment_ocid,
            auth_profile=self.config.oci.profile_name,
            context_size=gen_config.context_size
        )
        
        logger.info(f"Created LLM: {model_name}")
        return llm
    
    def create_reranker(self, top_n: int = None):
        """Create reranker if enabled"""
        if not self.config.reranker.enabled:
            return None
        
        if top_n is None:
            top_n = self.config.rag.top_n
        
        reranker = CohereRerank(
            api_key=self.config.reranker.api_key,
            top_n=top_n
        )
        
        logger.info(f"Created reranker with top_n={top_n}")
        return reranker
    
    def create_chat_engine(
        self,
        token_counter=None,
        verbose: bool = False,
        top_k: int = None,
        max_tokens: int = None,
        temperature: float = None,
        top_n: int = None
    ):
        """
        Create chat engine with RAG capabilities
        
        Args:
            token_counter: Token counting handler
            verbose: Enable verbose logging
            top_k: Number of documents to retrieve
            max_tokens: Maximum tokens for generation
            temperature: Temperature for generation
            top_n: Reranker top_n
        
        Returns:
            tuple: (chat_engine, token_counter)
        """
        logger.info("Creating chat engine...")
        print_configuration()
        
        # Use config defaults if not provided
        if top_k is None:
            top_k = self.config.rag.top_k
        if max_tokens is None:
            max_tokens = self.config.rag.max_tokens
        if temperature is None:
            temperature = self.config.rag.temperature
        if top_n is None:
            top_n = self.config.rag.top_n
        
        # Setup Phoenix tracing if enabled
        phoenix_config = self.config.observability.get('phoenix', {})
        if phoenix_config.get('enabled', False):
            try:
                import phoenix as px
                from llama_index.core.callbacks.global_handlers import set_global_handler
                
                os.environ["PHOENIX_PORT"] = phoenix_config.get('port', '7777')
                os.environ["PHOENIX_HOST"] = phoenix_config.get('host', '0.0.0.0')
                px.launch_app()
                set_global_handler("arize_phoenix")
                logger.info("Phoenix tracing enabled")
            except ImportError:
                logger.warning("Phoenix tracing enabled but dependencies not installed")
        
        # Create components
        embed_model = self.create_embedding_model()
        vector_store = OracleVectorStore(
            verbose=verbose,
            enable_hnsw_indexes=self.config.rag.enable_approximate
        )
        llm = self.create_llm()
        
        # Initialize tokenizer and token counter
        tokenizer = Tokenizer.from_pretrained(self.config.embedding_model.tokenizer)
        if token_counter is None:
            token_counter = TokenCountingHandler(tokenizer=tokenizer.encode)
        
        # Configure settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.callback_manager = CallbackManager([token_counter])
        
        # Create index and memory
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.config.rag.memory_token_limit,
            tokenizer_fn=tokenizer.encode
        )
        
        # Setup node postprocessors
        node_postprocessors = None
        if self.config.reranker.enabled:
            reranker = self.create_reranker(top_n=top_n)
            node_postprocessors = [reranker]
        
        # Create chat engine
        chat_engine = index.as_chat_engine(
            chat_mode=self.config.rag.chat_mode,
            memory=memory,
            verbose=verbose,
            similarity_top_k=top_k,
            node_postprocessors=node_postprocessors,
            streaming=self.config.rag.stream,
            system_prompt=self.config.rag.system_prompt,
        )
        
        logger.info("Chat engine created successfully")
        return chat_engine, token_counter
    
    def llm_chat(self, question: str, model_name: str = None) -> str:
        """
        Direct LLM chat without RAG
        
        Args:
            question: User question
            model_name: Optional model override
        
        Returns:
            LLM response
        """
        logger.info("Calling LLM chat (no RAG)...")
        
        llm = self.create_llm(model_name)
        response = llm.chat([ChatMessage(role="user", content=question)])
        
        logger.info("Response generated")
        return response.message.content


# Global instance
_chat_engine_manager = None


def get_chat_engine_manager() -> ChatEngineManager:
    """Get or create ChatEngineManager instance"""
    global _chat_engine_manager
    if _chat_engine_manager is None:
        _chat_engine_manager = ChatEngineManager()
    return _chat_engine_manager


def create_chat_engine(*args, **kwargs):
    """Convenience function to create chat engine"""
    manager = get_chat_engine_manager()
    return manager.create_chat_engine(*args, **kwargs)


def llm_chat(question: str, model_name: str = None) -> str:
    """Convenience function for LLM chat"""
    manager = get_chat_engine_manager()
    return manager.llm_chat(question, model_name)
