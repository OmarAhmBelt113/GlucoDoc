import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from src.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


logger = logging.getLogger(__name__)


def create_chunks(
    documents: list,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split parsed PDF documents into chunks using
    RecursiveCharacterTextSplitter.

    The splitter preserves natural Markdown boundaries
    whenever possible.
    """

    if not documents:
        logger.warning(
            "No documents provided for chunking."
        )
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    langchain_documents = [
        Document(
            page_content=document.text,
            metadata=document.metadata.copy(),
        )
        for document in documents
    ]

    chunks = splitter.split_documents(
        langchain_documents
    )

    # Add a stable chunk index per source page
    chunk_counters = {}

    for chunk in chunks:

        source = chunk.metadata.get(
            "source",
            "unknown",
        )

        page = chunk.metadata.get(
            "page",
            0,
        )

        key = (
            source,
            page,
        )

        chunk_index = chunk_counters.get(
            key,
            0,
        )

        chunk.metadata["chunk_index"] = (
            chunk_index
        )

        chunk_counters[key] = (
            chunk_index + 1
        )

    logger.info(
        "Created %d chunks from %d documents.",
        len(chunks),
        len(documents),
    )

    return chunks