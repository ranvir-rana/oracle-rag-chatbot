"""
Oracle Vector Store Module
Integrates Oracle Vector DB with llama-index
"""

import time
import array
import logging
import streamlit as st
from typing import List, Any, Dict
from contextlib import contextmanager

from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.schema import TextNode, BaseNode

from config_loader import get_config
from database import get_db_manager

logger = logging.getLogger(__name__)


@contextmanager
def optional_tracing(span_name: str):
    """Context manager for optional Phoenix tracing"""
    config = get_config()
    phoenix_config = config.observability.get('phoenix', {})
    
    if phoenix_config.get('enabled', False):
        try:
            from opentelemetry import trace as trace_api
            from opentelemetry.trace import Status, StatusCode
            from openinference.semconv.trace import SpanAttributes
            
            tracer = trace_api.get_tracer(__name__)
            with tracer.start_as_current_span(name=span_name) as span:
                span.set_attribute("openinference.span.kind", "Retriever")
                span.set_attribute(SpanAttributes.TOOL_NAME, "oracle_vector_store")
                span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, "Oracle DB 23ai")
                span.set_status(Status(StatusCode.OK))
                yield span
        except ImportError:
            logger.warning("Phoenix tracing enabled but dependencies not installed")
            yield None
    else:
        yield None


def oracle_query(
    embed_query: List[float],
    top_k: int,
    similarity_threshold: float = 0.35,
    verbose: bool = False,
    approximate: bool = False
) -> VectorStoreQueryResult:
    """
    Execute vector similarity query against Oracle Database
    
    Args:
        embed_query: Query embedding vector
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score
        verbose: Enable verbose logging
        approximate: Use approximate (HNSW) search
    
    Returns:
        VectorStoreQueryResult with matched chunks
    """
    start_time = time.time()
    config = get_config()
    db_manager = get_db_manager()
    
    try:
        with db_manager.get_connection() as connection:
            cursor = connection.cursor()
            
            # Prepare query vector
            array_type = "d" if config.embedding_model.bits == 64 else "f"
            array_query = array.array(array_type, embed_query)
            
            # Build SQL query
            approx_clause = "APPROXIMATE" if approximate else ""
            query_sql = f"""
                SELECT C.ID,
                       C.CHUNK,
                       C.PAGE_NUM,
                       VECTOR_DISTANCE(C.VEC, :1, COSINE) as distance,
                       B.NAME
                FROM CHUNKS C, BOOKS B
                WHERE C.BOOK_ID = B.ID
                ORDER BY distance
                FETCH {approx_clause} FIRST :2 ROWS ONLY
            """
            
            if verbose:
                logger.debug(f"Executing query with top_k={top_k}, threshold={similarity_threshold}")
            
            # Execute query
            cursor.execute(query_sql, [array_query, top_k])
            rows = cursor.fetchall()
            
            # Process results
            result_nodes = []
            node_ids = []
            similarities = []
            
            for row in rows:
                similarity_score = 1 - row[3]  # Convert distance to similarity
                
                if similarity_score >= similarity_threshold:
                    clob_data = row[1].read()
                    
                    result_nodes.append(
                        TextNode(
                            id_=row[0],
                            text=clob_data,
                            metadata={
                                "file_name": row[4],
                                "page#": row[2],
                                "Similarity Score": similarity_score
                            }
                        )
                    )
                    node_ids.append(row[0])
                    similarities.append(row[3])
                    
                    if verbose:
                        logger.debug(f"Added result: {row[0]}, similarity: {similarity_score:.4f}")
            
            cursor.close()
            
            elapsed_time = time.time() - start_time
            if verbose:
                logger.info(f"Query completed in {elapsed_time:.2f}s, found {len(result_nodes)} results")
            
            return VectorStoreQueryResult(
                nodes=result_nodes,
                similarities=similarities,
                ids=node_ids
            )
            
    except Exception as e:
        logger.error(f"Error in oracle_query: {e}")
        return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])


class OracleVectorStore(VectorStore):
    """Oracle Database Vector Store for llama-index"""
    
    stores_text: bool = True
    verbose: bool = False
    
    def __init__(self, verbose: bool = False, enable_hnsw_indexes: bool = False):
        """
        Initialize Oracle Vector Store
        
        Args:
            verbose: Enable verbose logging
            enable_hnsw_indexes: Use HNSW indexes for approximate search
        """
        self.verbose = verbose
        self.enable_hnsw_indexes = enable_hnsw_indexes
        self.node_dict: Dict[str, BaseNode] = {}
        
        logger.info("Initialized Oracle Vector Store")
    
    def add(self, nodes: List[BaseNode]) -> List[str]:
        """
        Add nodes to the index
        
        Args:
            nodes: List of nodes to add
        
        Returns:
            List of node IDs
        """
        ids_list = []
        for node in nodes:
            self.node_dict[node.id_] = node
            ids_list.append(node.id_)
        
        return ids_list
    
    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete node from index"""
        raise NotImplementedError("Delete not implemented for Oracle Vector Store")
    
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query the vector store
        
        Args:
            query: Vector store query object
        
        Returns:
            Query results
        """
        config = get_config()
        
        # Get parameters from session state or config
        if hasattr(st, 'session_state') and 'top_k' in st.session_state:
            top_k = st.session_state['top_k']
            similarity_threshold = st.session_state.get('similarity', config.rag.similarity_threshold)
        else:
            top_k = config.rag.top_k
            similarity_threshold = config.rag.similarity_threshold
        
        if self.verbose:
            logger.info(f"Querying DB with top_k={top_k}")
        
        with optional_tracing("oracle_vector_db"):
            return oracle_query(
                query.query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                verbose=self.verbose,
                approximate=self.enable_hnsw_indexes
            )
    
    def persist(self, persist_path=None, fs=None) -> None:
        """Persist vector store (already persisted in DB)"""
        logger.info("Vector store persistence handled by Oracle Database")
