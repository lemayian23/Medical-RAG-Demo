"""
Ingest Documents Script - Build FAISS Index from Medical Documents with OCR Support
"""

import os
import sys
import json
import io
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import config
from app.core.embeddings import MedCPTEmbeddings
from app.core.vectorstore import FAISSVectorStore
from app.core.ocr import get_ocr
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_documents_with_ocr(raw_dir: str) -> List[Dict]:
    """
    Load all documents from raw_dir, using OCR when needed.
    
    Supported formats:
    - .txt: Direct text
    - .pdf: Digital extraction first, OCR fallback
    - .jpg, .png, .jpeg: OCR
    - .docx: python-docx extraction
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    documents = []
    ocr = get_ocr()
    
    # Supported extensions
    text_extensions = ['.txt']
    pdf_extensions = ['.pdf']
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    docx_extensions = ['.docx']
    
    all_extensions = text_extensions + pdf_extensions + image_extensions + docx_extensions
    
    # Get all files
    files = []
    for ext in all_extensions:
        files.extend(list(raw_path.glob(f"*{ext}")))
    
    if not files:
        logger.warning("No files found in raw directory")
        return documents
    
    logger.info(f"Found {len(files)} files to process")
    
    for file_path in files:
        try:
            logger.info(f"Processing: {file_path.name}")
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Process with OCR
            result = ocr.process_document(file_data, file_path.name)
            
            if result["success"]:
                documents.append({
                    "text": result["text"],
                    "source": file_path.name,
                    "ocr_used": result.get("ocr_used", False),
                    "structured_data": result.get("structured_data", {}),
                })
                logger.info(f"✅ Loaded: {file_path.name} ({len(result['text'])} chars, OCR: {result.get('ocr_used', False)})")
            else:
                logger.error(f"❌ Failed: {file_path.name} - {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
    
    return documents


def create_chunks(text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Split text into chunks with metadata.
    """
    chunks = []
    text_len = len(text)
    
    if text_len == 0:
        return chunks
    
    for start in range(0, text_len, chunk_size - overlap):
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        
        if len(chunk) > 50:  # Only keep meaningful chunks
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": len(chunks),
                "start": start,
                "end": end,
            })
    
    return chunks


def main():
    """Main ingestion function."""
    logger.info("=" * 60)
    logger.info("📥 MEDICAL RAG - DOCUMENT INGESTION (with OCR)")
    logger.info("=" * 60)
    
    # 1. Load documents with OCR
    raw_dir = "./data/raw"
    documents = load_documents_with_ocr(raw_dir)
    
    if not documents:
        logger.error("❌ No documents loaded. Please add files to ./data/raw/")
        return
    
    logger.info(f"📄 Total documents: {len(documents)}")
    
    # 2. Create chunks
    all_chunks = []
    for doc in documents:
        chunks = create_chunks(
            doc["text"],
            doc["source"],
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        all_chunks.extend(chunks)
        logger.info(f"   {doc['source']}: {len(chunks)} chunks")
    
    if not all_chunks:
        logger.error("❌ No chunks created. Check your documents.")
        return
    
    logger.info(f"📦 Total chunks created: {len(all_chunks)}")
    
    # 3. Extract texts and sources
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_sources = [chunk["source"] for chunk in all_chunks]
    
    # 4. Initialize embeddings
    logger.info("🧠 Initializing embeddings...")
    embeddings = MedCPTEmbeddings()
    
    # 5. Generate embeddings
    logger.info("🔢 Generating chunk embeddings...")
    chunk_vectors = embeddings.encode_documents(chunk_texts)
    
    # 6. Initialize vector store
    logger.info("💾 Initializing FAISS vector store...")
    vectorstore = FAISSVectorStore(
        embedding_dim=384,
        index_path=config.FAISS_INDEX_PATH,
        metadata_path=config.METADATA_PATH,
    )
    
    # 7. Add chunks
    logger.info("📥 Adding chunks to vector store...")
    vectorstore.add_chunks(
        texts=chunk_texts,
        embeddings=chunk_vectors,
        sources=chunk_sources,
    )
    
    # 8. Save
    vectorstore.save()
    
    # 9. Display stats
    stats = vectorstore.get_stats()
    logger.info("=" * 60)
    logger.info("✅ INGESTION COMPLETE!")
    logger.info(f"📊 Total vectors: {stats['total_vectors']}")
    logger.info(f"📂 Metadata count: {stats['metadata_count']}")
    logger.info(f"📁 Index saved to: {config.FAISS_INDEX_PATH}")
    
    # 10. Show OCR summary
    ocr_count = sum(1 for d in documents if d.get("ocr_used", False))
    if ocr_count > 0:
        logger.info(f"📄 OCR used for: {ocr_count} documents")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()