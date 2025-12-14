"""
Configuration Management Module
Handles loading and validation of configuration from YAML and environment variables
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    user: str
    password: str
    dsn: str
    host_ip: str
    service: str
    wallet_location: str
    wallet_password: str
    config_dir: str


@dataclass
class OCIConfig:
    """OCI configuration"""
    compartment_ocid: str
    endpoint: str
    profile_name: str


@dataclass
class EmbeddingModelConfig:
    """Embedding model configuration"""
    type: str
    model_name: str
    tokenizer: str
    truncate: str
    bits: int


@dataclass
class GenerationModelConfig:
    """Generation model configuration"""
    type: str
    default_model: str
    available_models: list
    context_size: int


@dataclass
class RerankerConfig:
    """Reranker configuration"""
    enabled: bool
    type: str
    api_key: str
    model_id: str


@dataclass
class RAGConfig:
    """RAG pipeline configuration"""
    chunking_enabled: bool
    max_chunk_size: int
    chunk_overlap: int
    top_k: int
    top_n: int
    similarity_threshold: float
    enable_approximate: bool
    max_tokens: int
    temperature: float
    stream: bool
    chat_mode: str
    memory_token_limit: int
    system_prompt: str


@dataclass
class DocumentConfig:
    """Document processing configuration"""
    upload_dir: str
    processed_dir: str
    batch_size: int
    supported_formats: list
    id_generation_method: str


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    format: str
    file: str
    console: bool


@dataclass
class AppConfig:
    """Application configuration"""
    title: str
    page_title: str
    layout: str
    port: int
    enable_cors: bool


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load environment variables
        load_dotenv()
        
        # Load YAML configuration
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self._raw_config = yaml.safe_load(f)
        
        # Parse configurations
        self.app = self._parse_app_config()
        self.database = self._parse_database_config()
        self.oci = self._parse_oci_config()
        self.embedding_model = self._parse_embedding_model_config()
        self.generation_model = self._parse_generation_model_config()
        self.reranker = self._parse_reranker_config()
        self.rag = self._parse_rag_config()
        self.documents = self._parse_document_config()
        self.logging = self._parse_logging_config()
        
        # Additional settings
        self.features = self._raw_config.get('features', {})
        self.ui = self._raw_config.get('ui', {})
        self.observability = self._raw_config.get('observability', {})
    
    def _resolve_env_var(self, value: str) -> str:
        """Resolve environment variable placeholders like ${VAR_NAME}"""
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            resolved = os.getenv(env_var)
            if resolved is None:
                raise ValueError(f"Environment variable {env_var} not set")
            return resolved
        return value
    
    def _parse_app_config(self) -> AppConfig:
        """Parse application configuration"""
        app_cfg = self._raw_config.get('app', {})
        return AppConfig(
            title=app_cfg.get('title', 'AI Assistant'),
            page_title=app_cfg.get('page_title', 'AI Assistant'),
            layout=app_cfg.get('layout', 'centered'),
            port=app_cfg.get('port', 8501),
            enable_cors=app_cfg.get('enable_cors', False)
        )
    
    def _parse_database_config(self) -> DatabaseConfig:
        """Parse database configuration"""
        db_cfg = self._raw_config.get('database', {})
        wallet_cfg = db_cfg.get('wallet', {})
        
        return DatabaseConfig(
            user=db_cfg.get('user', 'ADMIN'),
            password=self._resolve_env_var(db_cfg.get('password', '')),
            dsn=db_cfg.get('dsn', ''),
            host_ip=db_cfg.get('host_ip', ''),
            service=db_cfg.get('service', ''),
            wallet_location=wallet_cfg.get('location', ''),
            wallet_password=self._resolve_env_var(wallet_cfg.get('password', '')),
            config_dir=wallet_cfg.get('config_dir', '')
        )
    
    def _parse_oci_config(self) -> OCIConfig:
        """Parse OCI configuration"""
        oci_cfg = self._raw_config.get('oci', {})
        return OCIConfig(
            compartment_ocid=self._resolve_env_var(oci_cfg.get('compartment_ocid', '')),
            endpoint=oci_cfg.get('endpoint', ''),
            profile_name=oci_cfg.get('profile_name', 'DEFAULT')
        )
    
    def _parse_embedding_model_config(self) -> EmbeddingModelConfig:
        """Parse embedding model configuration"""
        embed_cfg = self._raw_config.get('models', {}).get('embedding', {})
        return EmbeddingModelConfig(
            type=embed_cfg.get('type', 'OCI'),
            model_name=embed_cfg.get('model_name', ''),
            tokenizer=embed_cfg.get('tokenizer', ''),
            truncate=embed_cfg.get('truncate', 'END'),
            bits=embed_cfg.get('bits', 64)
        )
    
    def _parse_generation_model_config(self) -> GenerationModelConfig:
        """Parse generation model configuration"""
        gen_cfg = self._raw_config.get('models', {}).get('generation', {})
        return GenerationModelConfig(
            type=gen_cfg.get('type', 'OCI'),
            default_model=gen_cfg.get('default_model', ''),
            available_models=gen_cfg.get('available_models', []),
            context_size=gen_cfg.get('context_size', 128000)
        )
    
    def _parse_reranker_config(self) -> RerankerConfig:
        """Parse reranker configuration"""
        rerank_cfg = self._raw_config.get('models', {}).get('reranker', {})
        return RerankerConfig(
            enabled=rerank_cfg.get('enabled', False),
            type=rerank_cfg.get('type', 'COHERE'),
            api_key=self._resolve_env_var(rerank_cfg.get('api_key', '')),
            model_id=rerank_cfg.get('model_id', '')
        )
    
    def _parse_rag_config(self) -> RAGConfig:
        """Parse RAG configuration"""
        rag_cfg = self._raw_config.get('rag', {})
        chunking = rag_cfg.get('chunking', {})
        retrieval = rag_cfg.get('retrieval', {})
        generation = rag_cfg.get('generation', {})
        chat = rag_cfg.get('chat', {})
        
        return RAGConfig(
            chunking_enabled=chunking.get('enabled', True),
            max_chunk_size=chunking.get('max_chunk_size', 1000),
            chunk_overlap=chunking.get('chunk_overlap', 100),
            top_k=retrieval.get('top_k', 3),
            top_n=retrieval.get('top_n', 3),
            similarity_threshold=retrieval.get('similarity_threshold', 0.35),
            enable_approximate=retrieval.get('enable_approximate', False),
            max_tokens=generation.get('max_tokens', 600),
            temperature=generation.get('temperature', 0.1),
            stream=generation.get('stream', False),
            chat_mode=chat.get('mode', 'context'),
            memory_token_limit=chat.get('memory_token_limit', 3000),
            system_prompt=chat.get('system_prompt', '')
        )
    
    def _parse_document_config(self) -> DocumentConfig:
        """Parse document configuration"""
        doc_cfg = self._raw_config.get('documents', {})
        return DocumentConfig(
            upload_dir=doc_cfg.get('upload_dir', 'data/unprocessed'),
            processed_dir=doc_cfg.get('processed_dir', 'data/processed'),
            batch_size=doc_cfg.get('batch_size', 40),
            supported_formats=doc_cfg.get('supported_formats', ['pdf', 'txt']),
            id_generation_method=doc_cfg.get('id_generation_method', 'HASH')
        )
    
    def _parse_logging_config(self) -> LoggingConfig:
        """Parse logging configuration"""
        log_cfg = self._raw_config.get('logging', {})
        return LoggingConfig(
            level=log_cfg.get('level', 'INFO'),
            format=log_cfg.get('format', '%(asctime)s - %(message)s'),
            file=log_cfg.get('file', 'app.log'),
            console=log_cfg.get('console', True)
        )
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
        
        handlers = []
        if self.logging.console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(console_handler)
        
        if self.logging.file:
            file_handler = logging.FileHandler(self.logging.file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=log_level,
            format=self.logging.format,
            handlers=handlers
        )


# Global configuration instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
        _config_instance.setup_logging()
    return _config_instance


def reload_config(config_path: str = "config.yaml") -> Config:
    """Reload configuration"""
    global _config_instance
    _config_instance = Config(config_path)
    _config_instance.setup_logging()
    return _config_instance
