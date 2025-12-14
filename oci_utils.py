"""
OCI Utilities Module
Handles OCI configuration and client initialization
"""

import oci
import logging
from config_loader import get_config

logger = logging.getLogger(__name__)


def load_oci_config() -> dict:
    """
    Load OCI configuration from file
    
    Returns:
        dict: OCI configuration
    """
    config = get_config()
    profile_name = config.oci.profile_name
    
    try:
        oci_config = oci.config.from_file("~/.oci/config", profile_name)
        logger.info(f"Loaded OCI config with profile: {profile_name}")
        return oci_config
    except Exception as e:
        logger.error(f"Error loading OCI config: {e}")
        raise


def print_configuration():
    """Print current configuration for debugging"""
    config = get_config()
    
    logger.info("=" * 60)
    logger.info("Current Configuration:")
    logger.info("-" * 60)
    logger.info(f"Embedding Model: {config.embedding_model.type} - {config.embedding_model.model_name}")
    logger.info("Vector Store: Oracle AI Vector Search")
    logger.info(f"Generation Model: {config.generation_model.type}")
    logger.info(f"Default LLM: {config.generation_model.default_model}")
    logger.info("-" * 60)
    logger.info("Retrieval Parameters:")
    logger.info(f"  TOP_K: {config.rag.top_k}")
    logger.info(f"  Similarity Threshold: {config.rag.similarity_threshold}")
    
    if config.reranker.enabled:
        logger.info(f"Reranker: {config.reranker.type}")
        logger.info(f"  TOP_N: {config.rag.top_n}")
    
    if config.observability.get('phoenix', {}).get('enabled', False):
        logger.info("Observability: Phoenix Tracing Enabled")
    
    logger.info("=" * 60)
