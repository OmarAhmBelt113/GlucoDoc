import logging

from langchain_huggingface import HuggingFaceEmbeddings

from src.settings import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
)


logger = logging.getLogger(__name__)


def create_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    device: str = EMBEDDING_DEVICE,
) -> HuggingFaceEmbeddings:
    """
    Create the Hugging Face embedding model.

    If the model is not available in the local
    Hugging Face cache, it will be downloaded automatically.
    """

    logger.info("=" * 60)
    logger.info("Loading embedding model")
    logger.info("Model : %s", model_name)
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    logger.info(
        "Checking local Hugging Face cache..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    logger.info("Embedding model loaded successfully.")

    return embeddings