"""
Database Utilities Module
Handles Oracle Database connections and operations
"""

import oracledb
import logging
from typing import Optional, List, Tuple
from contextlib import contextmanager
from config_loader import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages Oracle Database connections and operations"""
    
    def __init__(self):
        self.config = get_config()
        self.db_config = self.config.database
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager"""
        connection = None
        try:
            connection = oracledb.connect(
                user=self.db_config.user,
                password=self.db_config.password,
                dsn=self.db_config.dsn,
                config_dir=self.db_config.config_dir,
                wallet_location=self.db_config.wallet_location,
                wallet_password=self.db_config.wallet_password
            )
            logger.info("Database connection established")
            yield connection
        except oracledb.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                try:
                    connection.close()
                    logger.debug("Database connection closed")
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
    
    def get_existing_documents(self) -> set:
        """Get list of existing document names from database"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT DISTINCT name FROM books")
                document_names = {name[0] for name in cursor.fetchall()}
                cursor.close()
                return document_names
        except oracledb.Error as e:
            logger.error(f"Error fetching existing documents: {e}")
            return set()
    
    def register_document(self, document_name: str) -> int:
        """Register a new document in the database"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # Get next ID
                cursor.execute("SELECT MAX(ID) FROM BOOKS")
                row = cursor.fetchone()
                new_id = (row[0] + 1) if row[0] is not None else 1
                
                # Insert new book
                cursor.execute(
                    "INSERT INTO BOOKS (ID, NAME) VALUES (:1, :2)",
                    [new_id, document_name]
                )
                connection.commit()
                cursor.close()
                
                logger.info(f"Registered document: {document_name} with ID: {new_id}")
                return new_id
        except oracledb.Error as e:
            logger.error(f"Error registering document: {e}")
            raise
    
    def save_chunks(
        self,
        chunk_ids: List[str],
        chunk_texts: List[str],
        page_nums: List[str],
        embeddings: List[List[float]],
        document_id: int
    ) -> int:
        """Save document chunks and embeddings to database"""
        import array
        errors = 0
        
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.setinputsizes(None, oracledb.DB_TYPE_CLOB)
                
                array_type = "d" if self.config.embedding_model.bits == 64 else "f"
                
                for chunk_id, text, page_num, vector in zip(
                    chunk_ids, chunk_texts, page_nums, embeddings
                ):
                    try:
                        input_array = array.array(array_type, vector)
                        cursor.execute(
                            """INSERT INTO CHUNKS (ID, CHUNK, VEC, PAGE_NUM, BOOK_ID) 
                               VALUES (:1, :2, :3, :4, :5)""",
                            [chunk_id, text, input_array, page_num, document_id]
                        )
                    except oracledb.Error as e:
                        logger.error(f"Error saving chunk {chunk_id}: {e}")
                        errors += 1
                
                connection.commit()
                cursor.close()
                logger.info(f"Saved {len(chunk_ids) - errors} chunks with {errors} errors")
                
        except oracledb.Error as e:
            logger.error(f"Critical error saving chunks: {e}")
            raise
        
        return errors


# Global instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get or create DatabaseManager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
