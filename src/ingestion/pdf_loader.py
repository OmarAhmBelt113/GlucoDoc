import logging
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
)

from src.settings import PDF_DIRECTORY


logger = logging.getLogger(__name__)


def load_pdfs(
    pdf_directory: Path = PDF_DIRECTORY,
):
    """Load all PDF pages from the specified directory."""

    if not pdf_directory.exists():
        raise FileNotFoundError(
            f"PDF directory does not exist: {pdf_directory}"
        )

    if not pdf_directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, got: {pdf_directory}"
        )

    loader = DirectoryLoader(
        str(pdf_directory),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )

    documents = loader.load()

    logger.info(
        "Loaded %d pages from %s",
        len(documents),
        pdf_directory,
    )

    return documents


if __name__ == "__main__":
    documents = load_pdfs()
    print(f"Loaded {len(documents)} pages")