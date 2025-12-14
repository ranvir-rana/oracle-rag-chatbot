-- ============================================================================
-- Oracle 23ai RAG Chatbot - Table Creation Script
-- ============================================================================

-- Create BOOKS table to store document metadata
CREATE TABLE IF NOT EXISTS BOOKS (
    ID NUMBER PRIMARY KEY,
    NAME VARCHAR2(500) NOT NULL,
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create CHUNKS table to store document chunks and embeddings
CREATE TABLE IF NOT EXISTS CHUNKS (
    ID VARCHAR2(64) PRIMARY KEY,
    CHUNK CLOB NOT NULL,
    VEC VECTOR,
    PAGE_NUM VARCHAR2(20),
    BOOK_ID NUMBER NOT NULL,
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_book FOREIGN KEY (BOOK_ID) REFERENCES BOOKS(ID) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON CHUNKS(BOOK_ID);
CREATE INDEX IF NOT EXISTS idx_books_name ON BOOKS(NAME);

-- Optional: Create HNSW index for approximate vector search (faster but less accurate)
-- Uncomment if you enable approximate search in config.yaml
-- CREATE VECTOR INDEX idx_chunks_vec_hnsw ON CHUNKS(VEC)
-- ORGANIZATION NEIGHBOR PARTITIONS
-- WITH DISTANCE COSINE
-- WITH TARGET ACCURACY 95;

COMMIT;
