"""
Document Processing Module
Handles document loading, chunking, embedding, and storage
"""

import os
import logging
import hashlib
import time
import shutil
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.oci_genai import OCIGenAIEmbeddings
from pypdf import PdfReader
from tokenizers import Tokenizer

from config_loader import get_config
from database import get_db_manager
from oci_utils import load_oci_config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing pipeline"""
    
    def __init__(self):
        self.config = get_config()
        self.db_manager = get_db_manager()
        
        # Initialize embedding model
        self.embed_model = OCIGenAIEmbeddings(
            compartment_id=self.config.oci.compartment_ocid,
            model_name=self.config.embedding_model.model_name,
            truncate=self.config.embedding_model.truncate,
            service_endpoint=self.config.oci.endpoint,
        )
        
        logger.info("Initialized Document Processor")
    
    def generate_ids(self, nodes: List[Document]) -> List[str]:
        """Generate IDs for document nodes"""
        method = self.config.documents.id_generation_method
        
        if method == "HASH":
            logger.debug("Generating hash-based IDs")
            ids = []
            for doc in tqdm(nodes, desc="Generating IDs"):
                hash_hex = hashlib.sha256(doc.text.encode()).hexdigest()
                ids.append(hash_hex)
            return ids
        elif method == "LLINDEX":
            return [doc.id_ for doc in nodes]
        else:
            raise ValueError(f"Unknown ID generation method: {method}")
    
    def load_pdf_pages(self, file_path: str) -> List[Document]:
        """Load PDF and extract pages"""
        pdf_reader = PdfReader(file_path)
        pages = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                pages.append(
                    Document(
                        text=text,
                        metadata={"page_label": str(page_num + 1)}
                    )
                )
        
        logger.info(f"Loaded {len(pages)} pages from PDF")
        return pages
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text content"""
        import re
        
        text = text.replace("\t", " ")
        text = text.replace(" -\n", "")
        text = text.replace("-\n", "")
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        return text if text else None
    
    def remove_short_pages(self, pages: List[Document], threshold: int = 10) -> List[Document]:
        """Remove pages with fewer words than threshold"""
        filtered = []
        removed = 0
        
        for page in pages:
            if len(page.text.split()) >= threshold:
                filtered.append(page)
            else:
                removed += 1
        
        logger.info(f"Removed {removed} short pages")
        return filtered
    
    def process_as_pages(self, file_path: str) -> Tuple[List[str], List[str], List[str]]:
        """Process document as pages (no chunking)"""
        pages = self.load_pdf_pages(file_path)
        
        # Preprocess text - create new Documents since text is immutable
        preprocessed_pages = []
        for page in pages:
            preprocessed_text = self.preprocess_text(page.text)
            if preprocessed_text:
                preprocessed_pages.append(
                    Document(text=preprocessed_text, metadata=page.metadata)
                )
        pages = preprocessed_pages
        
        pages = self.remove_short_pages(pages)
        
        pages_text = [p.text for p in pages]
        pages_num = [p.metadata["page_label"] for p in pages]
        pages_id = self.generate_ids(pages)
        
        return pages_text, pages_id, pages_num
    
    def process_as_chunks(self, file_path: str) -> Tuple[List[str], List[str], List[str]]:
        """Process document with chunking"""
        node_parser = SentenceSplitter(
            chunk_size=self.config.rag.max_chunk_size,
            chunk_overlap=self.config.rag.chunk_overlap
        )
        
        pages = self.load_pdf_pages(file_path)
        
        # Preprocess - create new Documents
        preprocessed_pages = []
        for page in pages:
            preprocessed_text = self.preprocess_text(page.text)
            if preprocessed_text and preprocessed_text.strip():
                preprocessed_pages.append(
                    Document(text=preprocessed_text, metadata=page.metadata)
                )
        pages = preprocessed_pages
        
        pages = self.remove_short_pages(pages)
        
        # Split into chunks
        nodes = node_parser.get_nodes_from_documents(pages, show_progress=True)
        
        nodes_text = [n.text for n in nodes]
        pages_num = [n.metadata.get("page_label", "unknown") for n in nodes]
        nodes_id = self.generate_ids(nodes)
        
        logger.info(f"Created {len(nodes)} chunks")
        return nodes_text, nodes_id, pages_num
    
    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for texts"""
        batch_size = self.config.documents.batch_size
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_model.get_text_embedding_batch(batch)
            embeddings.extend(batch_embeddings)
            time.sleep(0.1)  # Rate limiting
        
        logger.info(f"Computed {len(embeddings)} embeddings")
        return embeddings
    
    def process_document(self, file_path: str) -> dict:
        """
        Process a single document
        
        Returns:
            dict with processing statistics
        """
        start_time = time.time()
        document_name = os.path.basename(file_path)
        
        logger.info(f"Processing document: {document_name}")
        
        # Check if already processed
        existing_docs = self.db_manager.get_existing_documents()
        if document_name in existing_docs:
            logger.warning(f"Document {document_name} already exists in database")
            return {"status": "skipped", "reason": "already_exists"}
        
        # Process based on chunking setting
        if self.config.rag.chunking_enabled:
            logger.info(f"Chunking enabled (size: {self.config.rag.max_chunk_size})")
            texts, ids, page_nums = self.process_as_chunks(file_path)
        else:
            logger.info("Processing as pages")
            texts, ids, page_nums = self.process_as_pages(file_path)
        
        # Compute embeddings
        embeddings = self.compute_embeddings(texts)
        
        # Register document
        doc_id = self.db_manager.register_document(document_name)
        
        # Save to database
        errors = self.db_manager.save_chunks(ids, texts, page_nums, embeddings, doc_id)
        
        elapsed = time.time() - start_time
        
        stats = {
            "status": "success",
            "document_name": document_name,
            "chunks": len(texts),
            "errors": errors,
            "elapsed_time": elapsed
        }
        
        logger.info(f"Processed {document_name}: {len(texts)} chunks in {elapsed:.2f}s")
        return stats
    
    def process_directory(self, directory: str) -> dict:
        """Process all documents in a directory"""
        upload_dir = Path(directory)
        processed_dir = Path(self.config.documents.processed_dir)
        
        # Ensure directories exist
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Get files to process
        files = [
            f for f in upload_dir.glob("*")
            if f.is_file() and not f.name.startswith('.')
            and f.suffix[1:] in self.config.documents.supported_formats
        ]
        
        if not files:
            logger.warning(f"No files to process in {directory}")
            return {"status": "no_files"}
        
        logger.info(f"Found {len(files)} files to process")
        
        results = []
        total_chunks = 0
        
        for file_path in files:
            try:
                result = self.process_document(str(file_path))
                results.append(result)
                
                if result["status"] == "success":
                    total_chunks += result["chunks"]
                    # Move to processed directory
                    shutil.move(str(file_path), str(processed_dir / file_path.name))
                    logger.info(f"Moved {file_path.name} to processed directory")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                results.append({
                    "status": "error",
                    "document_name": file_path.name,
                    "error": str(e)
                })
        
        summary = {
            "total_files": len(files),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "errors": sum(1 for r in results if r["status"] == "error"),
            "total_chunks": total_chunks,
            "results": results
        }
        
        logger.info(f"Processing complete: {summary['successful']}/{summary['total_files']} successful")
        return summary


def main():
    """Main entry point for document processing"""
    config = get_config()
    processor = DocumentProcessor()
    
    upload_dir = config.documents.upload_dir
    
    logger.info("=" * 60)
    logger.info("Starting Document Processing")
    logger.info("=" * 60)
    
    summary = processor.process_directory(upload_dir)
    
    logger.info("=" * 60)
    logger.info("Processing Summary:")
    logger.info(f"  Total Files: {summary.get('total_files', 0)}")
    logger.info(f"  Successful: {summary.get('successful', 0)}")
    logger.info(f"  Skipped: {summary.get('skipped', 0)}")
    logger.info(f"  Errors: {summary.get('errors', 0)}")
    logger.info(f"  Total Chunks: {summary.get('total_chunks', 0)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()