import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import (
    ChatRequest,
    ChatResponse,
    EvidenceItem,
    Citation,
)

from src.embeddings.embedding_model import create_embedding_model
from src.rag.generator import create_llm
from src.rag.pipeline import run_rag
from src.vectorstore.chroma_store import load_vectorstore
from src.vectorstore.retriever import create_retriever


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("glucodoc.api")


# ============================================================
# RAG Components
# ============================================================

embedding_model = None
vectorstore = None
retriever = None
llm = None


# ============================================================
# RAG Initialization
# ============================================================

def initialize_rag():
    """
    Initialize all components required by the RAG pipeline.

    Components:
        1. Embedding model
        2. ChromaDB vector store
        3. Retriever
        4. LLM
    """

    global embedding_model
    global vectorstore
    global retriever
    global llm

    logger.info("=" * 70)
    logger.info("Initializing GlucoDoc RAG system")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. Embedding Model
    # --------------------------------------------------------

    logger.info(
        "Loading embedding model..."
    )

    embedding_model = create_embedding_model()

    logger.info(
        "Embedding model loaded successfully."
    )

    # --------------------------------------------------------
    # 2. ChromaDB
    # --------------------------------------------------------

    logger.info(
        "Loading ChromaDB..."
    )

    vectorstore = load_vectorstore(
        embedding_model=embedding_model,
    )

    logger.info(
        "ChromaDB loaded successfully."
    )

    # --------------------------------------------------------
    # 3. Retriever
    # --------------------------------------------------------

    logger.info(
        "Creating retriever..."
    )

    retriever = create_retriever(
        vectorstore=vectorstore,
    )

    logger.info(
        "Retriever created successfully."
    )

    # --------------------------------------------------------
    # 4. LLM
    # --------------------------------------------------------

    logger.info(
        "Loading LLM..."
    )

    llm = create_llm()

    logger.info(
        "LLM loaded successfully."
    )

    logger.info("=" * 70)
    logger.info(
        "GlucoDoc RAG system initialized successfully"
    )
    logger.info("=" * 70)


# ============================================================
# Application Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the RAG system when FastAPI starts.
    """

    try:

        initialize_rag()

        yield

    except Exception:

        logger.exception(
            "Failed to initialize GlucoDoc RAG system."
        )

        raise

    finally:

        logger.info(
            "Shutting down GlucoDoc API..."
        )


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="GlucoDoc API",
    description=(
        "RAG-powered API for diabetes-related "
        "medical information."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "name": "GlucoDoc",
        "description": "Diabetes RAG Assistant",
        "status": "online",
        "version": "1.0.0",
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():
    """
    Check whether the RAG system is initialized.
    """

    rag_initialized = all(
        component is not None
        for component in (
            embedding_model,
            vectorstore,
            retriever,
            llm,
        )
    )

    return {
        "status": (
            "healthy"
            if rag_initialized
            else "starting"
        ),
        "rag_initialized": rag_initialized,
    }


# ============================================================
# Chat Endpoint
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """
    Process a user question through the RAG pipeline.

    Returns:

        answer
        confidence
        evidence
        citations
    """

    # --------------------------------------------------------
    # Check RAG initialization
    # --------------------------------------------------------

    if (
        retriever is None
        or vectorstore is None
        or llm is None
    ):

        logger.error(
            "RAG system is not initialized."
        )

        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized.",
        )

    # --------------------------------------------------------
    # Validate Question
    # --------------------------------------------------------

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    logger.info(
        "Received question: %s",
        question,
    )

    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    try:

        result = run_rag(
            question=question,
            retriever=retriever,
            vectorstore=vectorstore,
            llm=llm,
            history=request.history,
            summarize=request.summarize,
        )

        # ----------------------------------------------------
        # Validate result
        # ----------------------------------------------------

        if not isinstance(
            result,
            dict,
        ):

            logger.error(
                "run_rag returned unexpected type: %s",
                type(result).__name__,
            )

            raise ValueError(
                "RAG pipeline must return a dictionary."
            )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        answer = str(
            result.get(
                "answer",
                "",
            )
        ).strip()

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        # ----------------------------------------------------
        # Evidence
        # ----------------------------------------------------

        evidence_items = []

        for item in result.get(
            "evidence",
            [],
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            evidence_items.append(
                EvidenceItem(
                    source=str(
                        item.get(
                            "source",
                            "Unknown source",
                        )
                    ),
                    page=str(
                        item.get(
                            "page",
                            "Unknown page",
                        )
                    ),
                    content=str(
                        item.get(
                            "content",
                            "",
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # Citations
        # ----------------------------------------------------

        citation_items = []

        for item in result.get(
            "citations",
            [],
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            citation_items.append(
                Citation(
                    source=str(
                        item.get(
                            "source",
                            "Unknown source",
                        )
                    ),
                    page=str(
                        item.get(
                            "page",
                            "Unknown page",
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # Final Response
        # ----------------------------------------------------

        response = ChatResponse(
            answer=answer,
            confidence=confidence,
            evidence=evidence_items,
            citations=citation_items,
        )

        logger.info(
            "Question processed successfully | "
            "confidence=%.4f | evidence=%d | citations=%d",
            confidence,
            len(evidence_items),
            len(citation_items),
        )

        return response

    # --------------------------------------------------------
    # HTTP Errors
    # --------------------------------------------------------

    except HTTPException:

        raise

    # --------------------------------------------------------
    # Unexpected Errors
    # --------------------------------------------------------

    except Exception:

        logger.exception(
            "Failed to process question."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to generate an answer. "
                "Please try again later."
            ),
        )
