import argparse
import logging

from src.ingestion.parser import parse_pdf_directory
from src.ingestion.chunker import create_chunks
from src.embeddings.embedding_model import create_embedding_model
from src.vectorstore.chroma_store import create_vectorstore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GlucoDoc Data Pipeline")
    parser.add_argument(
        "--ingest", 
        action="store_true", 
        help="Run the PDF parsing and ChromaDB ingestion pipeline"
    )
    args = parser.parse_args()

    if args.ingest:
        logger.info("Starting GlucoDoc ingestion pipeline...")

        # ==================================================
        # 1. Parse PDFs
        # ==================================================
        logger.info("Parsing PDF files...")
        documents = parse_pdf_directory()

        if not documents:
            logger.warning("No documents were parsed.")
            return

        logger.info("Total parsed documents/pages: %d", len(documents))

        # ==================================================
        # 2. Create chunks
        # ==================================================
        logger.info("Creating chunks...")
        chunks = create_chunks(documents)

        if not chunks:
            logger.warning("No chunks were created.")
            return

        logger.info("Total chunks created: %d", len(chunks))

        # ==================================================
        # 3. Load embedding model
        # ==================================================
        logger.info("Loading embedding model...")
        embedding_model = create_embedding_model()

        # ==================================================
        # 4. Create ChromaDB
        # ==================================================
        logger.info("Creating ChromaDB vector store...")
        vectorstore = create_vectorstore(
            chunks=chunks,
            embedding_model=embedding_model,
        )

        # ==================================================
        # 5. Verify database
        # ==================================================
        collection = vectorstore._collection
        logger.info("ChromaDB contains %d documents.", collection.count())

        print("\n" + "=" * 60)
        print("GLUCODOC INGESTION COMPLETED")
        print("=" * 60)
        print(f"Parsed documents/pages : {len(documents)}")
        print(f"Total chunks           : {len(chunks)}")
        print(f"ChromaDB documents     : {collection.count()}")
        print("=" * 60)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()