import logging

from langchain_core.documents import Document

from src.settings import RETRIEVAL_K


logger = logging.getLogger(__name__)


# ============================================================
# Retriever Creation
# ============================================================

def create_retriever(
    vectorstore,
    k: int = RETRIEVAL_K,
):
    """
    Create a similarity-based retriever from ChromaDB.

    The retriever is kept as a standard LangChain retriever
    so it remains compatible with the existing RAG pipeline.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    logger.info(
        "Creating retriever with k=%d",
        k,
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )


# ============================================================
# Retrieve Documents
# ============================================================

def retrieve_documents(
    retriever,
    query: str,
) -> list[Document]:
    """
    Retrieve relevant documents for a user query.

    This function preserves the existing interface used by
    the RAG pipeline.
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    query = query.strip()

    logger.info(
        "Retrieving documents for query: %s",
        query,
    )

    documents = retriever.invoke(
        query
    )

    logger.info(
        "Retrieved %d documents.",
        len(documents),
    )

    return list(documents)


# ============================================================
# Retrieve Documents With Scores
# ============================================================

def retrieve_documents_with_scores(
    vectorstore,
    query: str,
    k: int = RETRIEVAL_K,
) -> list[tuple[Document, float]]:
    """
    Retrieve documents together with their Chroma relevance
    scores.

    This function is used when the application needs an actual
    retrieval confidence score.

    Returns:

        [
            (Document, relevance_score),
            ...
        ]

    The relevance score is normalized by the vector store's
    relevance-score function and is expected to be higher for
    more relevant documents.
    """

    if not query or not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    if k <= 0:

        raise ValueError(
            "k must be greater than 0."
        )

    query = query.strip()

    logger.info(
        "Retrieving documents with scores for query: %s",
        query,
    )

    # --------------------------------------------------------
    # Chroma + LangChain relevance scores
    # --------------------------------------------------------

    results = (
        vectorstore
        .similarity_search_with_relevance_scores(
            query,
            k=k,
        )
    )

    scored_documents = []

    for document, score in results:

        try:

            score = float(score)

        except (
            TypeError,
            ValueError,
        ):

            score = 0.0

        # ----------------------------------------------------
        # Clamp score
        # ----------------------------------------------------

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        scored_documents.append(
            (
                document,
                score,
            )
        )

    logger.info(
        "Retrieved %d scored documents.",
        len(scored_documents),
    )

    if scored_documents:

        logger.info(
            "Top retrieval score: %.4f",
            scored_documents[0][1],
        )

    return scored_documents


# ============================================================
# Top Retrieval Confidence
# ============================================================

def get_retrieval_confidence(
    vectorstore,
    query: str,
    k: int = RETRIEVAL_K,
) -> float:
    """
    Calculate retrieval confidence from the top relevance score.

    The confidence is the highest relevance score among the
    retrieved documents.

    Returns:
        float between 0.0 and 1.0
    """

    results = retrieve_documents_with_scores(
        vectorstore=vectorstore,
        query=query,
        k=k,
    )

    if not results:

        return 0.0

    confidence = max(
        score
        for _, score in results
    )

    return round(
        confidence,
        4,
    )
