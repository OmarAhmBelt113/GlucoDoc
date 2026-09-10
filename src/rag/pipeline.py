import logging

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag.prompt import build_prompt
from src.rag.generator import generate_answer
from src.vectorstore.retriever import (
    retrieve_documents,
    retrieve_documents_with_scores,
)


logger = logging.getLogger(__name__)


# ============================================================
# Metadata Helpers
# ============================================================

def _get_source(document: Document) -> str:
    """
    Extract source filename from document metadata.
    """

    metadata = document.metadata or {}

    return str(
        metadata.get(
            "file_name",
            metadata.get(
                "source",
                "Unknown source",
            ),
        )
    )


def _get_page(document: Document) -> str:
    """
    Extract page number from document metadata.
    """

    metadata = document.metadata or {}

    return str(
        metadata.get(
            "page",
            metadata.get(
                "page_number",
                "Unknown page",
            ),
        )
    )


# ============================================================
# Evidence Builder
# ============================================================

def _build_evidence(
    scored_documents: list[tuple[Document, float]],
) -> list[dict]:
    """
    Convert retrieved documents into API-friendly evidence.

    Each evidence item contains:

        source
        page
        content
        score
    """

    evidence = []

    for document, score in scored_documents:

        evidence.append(
            {
                "source": _get_source(document),
                "page": _get_page(document),
                "content": (
                    document.page_content or ""
                ).strip(),
                "score": round(
                    float(score),
                    4,
                ),
            }
        )

    return evidence


# ============================================================
# Citation Builder
# ============================================================

def _build_citations(
    documents: list[Document],
) -> list[dict]:
    """
    Build unique source/page citations.

    Duplicate citations are removed while preserving order.
    """

    citations = []

    seen = set()

    for document in documents:

        source = _get_source(
            document
        )

        page = _get_page(
            document
        )

        key = (
            source,
            page,
        )

        if key in seen:
            continue

        seen.add(key)

        citations.append(
            {
                "source": source,
                "page": page,
            }
        )

    return citations


# ============================================================
# Confidence Calculation
# ============================================================

def _calculate_confidence(
    scored_documents: list[tuple[Document, float]],
) -> float:
    """
    Calculate retrieval confidence.

    The confidence is based on the highest relevance score
    returned by Chroma.

    Returns:
        float between 0.0 and 1.0
    """

    if not scored_documents:

        return 0.0

    scores = [
        float(score)
        for _, score in scored_documents
    ]

    confidence = max(
        scores
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    return round(
        confidence,
        4,
    )


# ============================================================
# Main RAG Pipeline
# ============================================================

def run_rag(
    question: str,
    retriever,
    vectorstore,
    llm: ChatGoogleGenerativeAI,
    history: list = None,
    summarize: bool = False,
) -> dict:
    """
    Run the complete GlucoDoc RAG pipeline.

    Pipeline:

        1. Retrieve relevant documents.
        2. Retrieve relevance scores.
        3. Calculate retrieval confidence.
        4. Build evidence.
        5. Build citations.
        6. Build LLM prompt.
        7. Generate answer.
        8. Return structured response.

    Returns:

        {
            "answer": str,
            "confidence": float,
            "evidence": list,
            "citations": list,
        }
    """

    # ========================================================
    # Validate Question
    # ========================================================

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    logger.info(
        "Running RAG pipeline for question: %s",
        question,
    )

    # ========================================================
    # 1. Retrieve Documents
    # ========================================================

    logger.info(
        "Retrieving relevant documents..."
    )

    documents = retrieve_documents(
        retriever=retriever,
        query=question,
    )

    logger.info(
        "Retrieved %d documents.",
        len(documents),
    )

    # ========================================================
    # 2. Retrieve Documents With Scores
    # ========================================================

    logger.info(
        "Calculating retrieval relevance scores..."
    )

    scored_documents = retrieve_documents_with_scores(
        vectorstore=vectorstore,
        query=question,
        k=len(documents),
    )

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if not scored_documents:

        scored_documents = [
            (
                document,
                0.0,
            )
            for document in documents
        ]

    logger.info(
        "Obtained %d scored documents.",
        len(scored_documents),
    )

    # ========================================================
    # 3. Calculate Confidence
    # ========================================================

    confidence = _calculate_confidence(
        scored_documents=scored_documents,
    )

    logger.info(
        "Retrieval confidence: %.4f",
        confidence,
    )

    if confidence < 0.25:
        logger.warning("Confidence is below safety threshold (0.25). Refusing to answer.")
        return {
            "answer": "The provided guideline documents do not contain sufficient evidence to answer this question securely.",
            "confidence": confidence,
            "evidence": _build_evidence(scored_documents),
            "citations": [],
        }

    # ========================================================
    # 4. Evidence
    # ========================================================

    evidence = _build_evidence(
        scored_documents=scored_documents,
    )

    logger.info(
        "Built %d evidence items.",
        len(evidence),
    )

    # ========================================================
    # 5. Citations
    # ========================================================

    citation_documents = [
        document
        for document, _ in scored_documents
    ]

    citations = _build_citations(
        documents=citation_documents,
    )

    logger.info(
        "Built %d citations.",
        len(citations),
    )

    # ========================================================
    # 6. Build Prompt
    # ========================================================

    logger.info(
        "Building LLM prompt..."
    )

    prompt = build_prompt(
        question=question,
        documents=documents,
        history=history,
        summarize=summarize,
    )

    # ========================================================
    # 7. Generate Answer
    # ========================================================

    logger.info(
        "Generating final answer..."
    )

    answer = generate_answer(
        llm=llm,
        prompt=prompt,
    )

    logger.info(
        "RAG answer generated successfully."
    )

    # ========================================================
    # 8. Structured Response
    # ========================================================

    result = {
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
        "citations": citations,
    }

    logger.info(
        "RAG pipeline completed successfully."
    )

    return result
