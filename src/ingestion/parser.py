import logging
import re
from pathlib import Path

from llama_parse import LlamaParse

from src.settings import (
    LLAMA_CLOUD_API_KEY,
    PDF_DIRECTORY,
)


logger = logging.getLogger(__name__)


def clean_text_content(text: str) -> str:
    """
    Clean parsed PDF text while preserving Markdown structure.

    Removes:
    - HTML formatting tags
    - HTML superscript tags
    - HTML line-break tags
    - standalone page numbers
    - known repeated headers/footers
    """

    if not text:
        return ""

    # ========================================================
    # Remove superscript HTML tags and their content
    # ========================================================

    text = re.sub(
        r"<sup[^>]*>.*?</sup>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # ========================================================
    # Replace HTML line breaks with spaces
    # ========================================================

    text = re.sub(
        r"<br\s*/?>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # ========================================================
    # Remove basic formatting tags
    # ========================================================

    text = re.sub(
        r"</?(b|i|u|strong|em)>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # ========================================================
    # Process lines
    # ========================================================

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:

        stripped_line = line.strip()

        # Preserve empty lines
        if not stripped_line:
            cleaned_lines.append("")
            continue

        # ====================================================
        # Remove standalone page numbers
        # ====================================================

        if stripped_line.isdigit() and len(stripped_line) <= 3:
            continue

        # ====================================================
        # Remove known headers / footers
        # ====================================================

        normalized_line = stripped_line.upper()

        if "HEARTS – D" in normalized_line:
            continue

        if "HEARTS - D" in normalized_line:
            continue

        if (
            "DEFINITION AND DIAGNOSIS OF DIABETES MELLITUS "
            "AND INTERMEDIATE HYPERGLYCEMIA"
            in normalized_line
        ):
            continue

        cleaned_lines.append(line)

    # ========================================================
    # Reconstruct text while preserving Markdown
    # ========================================================

    cleaned_text = "\n".join(cleaned_lines)

    # ========================================================
    # Remove excessive blank lines
    # ========================================================

    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_text,
    )

    return cleaned_text.strip()


def create_parser() -> LlamaParse:
    """
    Create and configure the LlamaParse parser.
    """

    if not LLAMA_CLOUD_API_KEY:
        raise EnvironmentError(
            "LLAMA_CLOUD_API_KEY is not set. "
            "Add it to the .env file."
        )

    logger.info(
        "Initializing LlamaParse..."
    )

    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        verbose=True,
        language="en",
    )

    logger.info(
        "LlamaParse initialized successfully."
    )

    return parser


def parse_pdf(
    file_path: Path,
    parser: LlamaParse | None = None,
):
    """
    Parse a single PDF file using LlamaParse.

    Args:
        file_path:
            Path to the PDF file.

        parser:
            Optional pre-configured LlamaParse instance.

    Returns:
        List of parsed page documents.
    """

    file_path = Path(file_path)

    # ========================================================
    # Validate file
    # ========================================================

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file does not exist: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {file_path}"
        )

    # ========================================================
    # Create parser if not provided
    # ========================================================

    parser = parser or create_parser()

    logger.info(
        "Parsing PDF: %s",
        file_path.name,
    )

    # ========================================================
    # Parse PDF
    # ========================================================

    documents = parser.load_data(
        str(file_path)
    )

    # ========================================================
    # Clean text + add metadata
    # ========================================================

    for page_number, document in enumerate(
        documents,
        start=1,
    ):

        cleaned_text = clean_text_content(
            document.text
        )

        document.set_content(
            cleaned_text
        )

        document.metadata.update(
            {
                "file_name": file_path.name,
                "source": str(file_path),
                "page": page_number,
                "file_type": "pdf",
            }
        )

    logger.info(
        "Parsed %d pages from %s",
        len(documents),
        file_path.name,
    )

    return documents


def parse_pdf_directory(
    pdf_directory: Path = PDF_DIRECTORY,
):
    """
    Parse all PDF files inside the configured PDF directory.

    Returns:
        List containing parsed documents from all PDFs.
    """

    pdf_directory = Path(pdf_directory)

    # ========================================================
    # Validate directory
    # ========================================================

    if not pdf_directory.exists():
        raise FileNotFoundError(
            f"PDF directory does not exist: {pdf_directory}"
        )

    if not pdf_directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, got: {pdf_directory}"
        )

    # ========================================================
    # Create one parser instance
    # ========================================================

    parser = create_parser()

    # ========================================================
    # Find PDFs
    # ========================================================

    pdf_files = sorted(
        pdf_directory.glob("*.pdf")
    )

    if not pdf_files:
        logger.warning(
            "No PDF files found in %s",
            pdf_directory,
        )
        return []

    logger.info(
        "Found %d PDF files in %s",
        len(pdf_files),
        pdf_directory,
    )

    # ========================================================
    # Parse all PDFs
    # ========================================================

    all_documents = []

    for pdf_file in pdf_files:

        try:

            documents = parse_pdf(
                pdf_file,
                parser=parser,
            )

            all_documents.extend(
                documents
            )

        except Exception:

            logger.exception(
                "Failed to parse PDF: %s",
                pdf_file.name,
            )

    # ========================================================
    # Final statistics
    # ========================================================

    logger.info(
        "Total parsed pages from %d PDFs: %d",
        len(pdf_files),
        len(all_documents),
    )

    return all_documents