import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from src.settings import (
    GOOGLE_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)


logger = logging.getLogger(__name__)


def create_llm() -> ChatGoogleGenerativeAI:
    """
    Create and configure the LLM used by GlucoDoc.
    """

    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Check your .env file."
        )

    logger.info(
        "Loading LLM: %s",
        LLM_MODEL_NAME,
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )

    logger.info(
        "LLM loaded successfully."
    )

    return llm


def generate_answer(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
) -> str:
    """
    Generate an answer using the provided LLM.
    """

    if not prompt or not prompt.strip():
        raise ValueError(
            "Prompt cannot be empty."
        )

    logger.info(
        "Generating answer..."
    )

    response = llm.invoke(prompt)

    content = response.content

    # Gemini may return structured content
    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))

        content = "".join(parts)

    elif isinstance(content, dict):
        content = content.get(
            "text",
            str(content),
        )

    else:
        content = str(content)

    return content.strip()