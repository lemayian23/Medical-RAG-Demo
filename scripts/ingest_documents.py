"""
Ingest Documents Script - Build FAISS Index from Medical Documents
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import config
from app.core.embeddings import MedCPTEmbeddings
from app.core.vectorstore import FAISSVectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_sample_documents() -> list:
    """
    Load sample medical documents.
    If no files exist, create sample documents.
    """
    raw_dir = Path("./data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if there are any files
    files = list(raw_dir.glob("*.*"))
    if files:
        logger.info(f"Loading {len(files)} documents from {raw_dir}")
        documents = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append({
                        "text": text,
                        "source": file_path.name,
                    })
                    logger.info(f"✅ Loaded: {file_path.name}")
            except Exception as e:
                logger.error(f"❌ Error loading {file_path.name}: {e}")
        return documents

    # No files found - create sample documents
    logger.info("No documents found. Creating sample medical documents...")

    sample_docs = [
        {
            "text": """
Breast Cancer Treatment Guidelines

HER2-positive breast cancer is an aggressive subtype that accounts for 15-20% of all breast cancers. 
Treatment options include:
1. Trastuzumab (Herceptin) - a monoclonal antibody targeting HER2 receptors
2. Pertuzumab - another HER2-directed monoclonal antibody
3. T-DM1 (Kadcyla) - an antibody-drug conjugate
4. Lapatinib - a small molecule tyrosine kinase inhibitor
5. Neratinib - an irreversible pan-HER tyrosine kinase inhibitor

First-line treatment for metastatic HER2-positive breast cancer typically combines trastuzumab, pertuzumab, and a taxane.
            """,
            "source": "breast_cancer_guidelines.txt"
        },
        {
            "text": """
Tamoxifen Mechanism of Action

Tamoxifen is a selective estrogen receptor modulator (SERM) used in the treatment of hormone receptor-positive breast cancer.

Mechanism: Tamoxifen competitively binds to estrogen receptors (ER) on breast cancer cells, blocking estrogen from binding. This prevents estrogen from stimulating tumor growth. In breast tissue, tamoxifen acts as an anti-estrogen. 

Indications:
- Adjuvant treatment of ER-positive breast cancer
- Metastatic breast cancer
- Ductal carcinoma in situ (DCIS)
- Breast cancer prevention in high-risk women

Side effects include hot flashes, vaginal discharge, and increased risk of endometrial cancer.
            """,
            "source": "tamoxifen_mechanism.txt"
        },
        {
            "text": """
Paclitaxel Side Effects

Paclitaxel (Taxol) is a chemotherapy drug used in the treatment of breast, ovarian, and lung cancers.

Common side effects:
1. Neutropenia (low white blood cell count) - most common dose-limiting toxicity
2. Peripheral neuropathy - numbness and tingling in hands and feet
3. Alopecia (hair loss)
4. Nausea and vomiting
5. Myalgia and arthralgia (muscle and joint pain)
6. Hypersensitivity reactions - can be severe, requiring premedication

Management: Premedication with corticosteroids, antihistamines, and H2 blockers is standard.
Neuropathy management may require dose reduction or treatment interruption.
            """,
            "source": "paclitaxel_side_effects.txt"
        },
    ]

    # Save sample documents
    for doc in sample_docs:
        file_path = raw_dir / doc["source"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["text"])
        logger.info(f"✅ Created: {file_path}")

    return sample_docs


def main():
    """Main ingestion function."""
    logger.info("=" * 60)
    logger.info("📥 MEDICAL RAG - DOCUMENT INGESTION")
    logger.info("=" * 60)

    # Load documents
    documents = load_sample_documents()

    if not documents:
        logger.error("No documents found. Please add files to ./data/raw/")
        return

    logger.info(f"📄 Total documents: {len(documents)}")

    # Extract text and sources
    texts = [doc["text"] for doc in documents]
    sources = [doc["source"] for doc in documents]

    # Initialize embeddings
    logger.info("🧠 Initializing MedCPT embeddings...")
    embeddings = MedCPTEmbeddings()

    # Create embeddings
    logger.info("🔢 Generating embeddings...")
    vectors = embeddings.encode_documents(texts)

    # Create chunks (simple approach - one chunk per document for demo)
    chunks = []
    chunk_sources = []
    chunk_texts = []

    for i, (text, source) in enumerate(zip(texts, sources)):
        # Split into chunks of ~500 chars with overlap
        chunk_size = 500
        overlap = 100
        text_len = len(text)

        for start in range(0, text_len, chunk_size - overlap):
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()
            if len(chunk) > 50:  # Only keep meaningful chunks
                chunks.append(chunk)
                chunk_sources.append(source)
                chunk_texts.append(chunk)

    logger.info(f"📦 Created {len(chunks)} chunks from {len(documents)} documents")

    if not chunks:
        logger.error("No chunks created. Check your documents.")
        return

    # Generate embeddings for chunks
    logger.info("🔢 Generating chunk embeddings...")
    chunk_vectors = embeddings.encode_documents(chunk_texts)

    # Initialize vector store
    logger.info("💾 Initializing FAISS vector store...")
    vectorstore = FAISSVectorStore(
        embedding_dim=384,
        index_path=config.FAISS_INDEX_PATH,
        metadata_path=config.METADATA_PATH,
    )

    # Add chunks
    logger.info("📥 Adding chunks to vector store...")
    vectorstore.add_chunks(
        texts=chunk_texts,
        embeddings=chunk_vectors,
        sources=chunk_sources,
    )

    # Save
    vectorstore.save()

    # Display stats
    stats = vectorstore.get_stats()
    logger.info("=" * 60)
    logger.info("✅ INGESTION COMPLETE!")
    logger.info(f"📊 Total vectors: {stats['total_vectors']}")
    logger.info(f"📂 Metadata count: {stats['metadata_count']}")
    logger.info(f"📁 Index saved to: {config.FAISS_INDEX_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()