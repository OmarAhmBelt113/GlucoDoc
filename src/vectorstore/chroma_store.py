import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIRECTORY,
)


logger = logging.getLogger(__name__)


def generate_chunk_id(
    chunk: Document,
) -> str:
    """
    Generate a stable unique ID for a chunk.

    The ID is based on:
    - source file
    - page number
    - chunk index
    """

    source = chunk.metadata.get(
        "source",
        "unknown",
    )

    page = chunk.metadata.get(
        "page",
        0,
    )

    chunk_index = chunk.metadata.get(
        "chunk_index",
        0,
    )

    raw_id = (
        f"{source}|"
        f"{page}|"
        f"{chunk_index}"
    )

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


def create_vectorstore(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: Path = CHROMA_DIRECTORY,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Create or update a persistent ChromaDB vector store.

    Running the ingestion pipeline multiple times with
    the same chunks will not create duplicates.
    """

    if not chunks:
        raise ValueError(
            "Cannot create ChromaDB from an empty chunk list."
        )

    persist_directory = Path(
        persist_directory
    )

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Opening ChromaDB at %s",
        persist_directory,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(
            persist_directory
        ),
    )

    ids = [
        generate_chunk_id(chunk)
        for chunk in chunks
    ]

    existing_ids = set(
        vectorstore.get(
            ids=ids
        )["ids"]
    )

    new_chunks = []
    new_ids = []

    for chunk, chunk_id in zip(
        chunks,
        ids,
    ):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:

        logger.info(
            "Adding %d new chunks to ChromaDB.",
            len(new_chunks),
        )

        vectorstore.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )

    else:

        logger.info(
            "No new chunks found. "
            "ChromaDB is already up to date."
        )

    logger.info(
        "ChromaDB collection contains %d documents.",
        vectorstore._collection.count(),
    )

    return vectorstore


def load_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: Path = CHROMA_DIRECTORY,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Load an existing persistent ChromaDB vector store.

    This function does not parse PDFs, create chunks,
    or generate new embeddings.
    """

    persist_directory = Path(
        persist_directory
    )

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"ChromaDB directory does not exist: "
            f"{persist_directory}"
        )

    logger.info(
        "Loading existing ChromaDB from %s",
        persist_directory,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(
            persist_directory
        ),
    )

    count = vectorstore._collection.count()

    if count == 0:
        raise ValueError(
            "ChromaDB collection exists but contains no documents."
        )

    logger.info(
        "Loaded ChromaDB with %d documents.",
        count,
    )

    return vectorstore