from pydantic import BaseModel, Field


# ============================================================
# Chat Request
# ============================================================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    """
    Request body for the /chat endpoint.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="User question",
        examples=[
            "What are the diagnostic criteria for diabetes mellitus?"
        ],
    )

    history: list[Message] = Field(
        default_factory=list,
        description="Conversation history",
    )

    summarize: bool = Field(
        default=False,
        description="If true, the system will provide a short summary instead of a detailed answer.",
    )


# ============================================================
# Evidence
# ============================================================

class EvidenceItem(BaseModel):
    """
    Retrieved evidence used by the RAG system.
    """

    source: str = Field(
        ...,
        description="Source document name",
    )

    page: str = Field(
        ...,
        description="Source page number",
    )

    content: str = Field(
        ...,
        description="Retrieved document content",
    )


# ============================================================
# Citation
# ============================================================

class Citation(BaseModel):
    """
    Citation information for retrieved evidence.
    """

    source: str = Field(
        ...,
        description="Source document name",
    )

    page: str = Field(
        ...,
        description="Source page number",
    )


# ============================================================
# Chat Response
# ============================================================

class ChatResponse(BaseModel):
    """
    Response returned by the /chat endpoint.
    """

    answer: str = Field(
        ...,
        description="Generated or refusal answer",
    )

    confidence: float = Field(
        ...,
        description=(
            "Top retrieval relevance score "
            "used by the safety gate"
        ),
    )

    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Retrieved evidence documents",
    )

    citations: list[Citation] = Field(
        default_factory=list,
        description="Source citations",
    )