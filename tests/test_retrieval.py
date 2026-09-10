import logging

from src.embeddings.embedding_model import create_embedding_model
from src.settings import RETRIEVAL_K
from src.vectorstore.chroma_store import load_vectorstore
from src.vectorstore.retriever import (
    create_retriever,
    retrieve_documents,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main():
    # ========================================================
    # 1. Load embedding model
    # ========================================================

    print("\nLoading embedding model...")

    embedding_model = create_embedding_model()

    # ========================================================
    # 2. Load existing ChromaDB
    # ========================================================

    print("Loading ChromaDB...")

    vectorstore = load_vectorstore(
        embedding_model=embedding_model,
    )

    # ========================================================
    # 3. Create retriever
    # ========================================================

    retriever = create_retriever(
        vectorstore=vectorstore,
        k=RETRIEVAL_K,
    )

    # ========================================================
    # 4. Test query
    # ========================================================

    query = (
        "What are the diagnostic criteria "
        "for diabetes mellitus?"
    )

    print("\n" + "=" * 80)
    print(f"QUERY:\n{query}")
    print("=" * 80)

    # ========================================================
    # 5. Retrieve documents
    # ========================================================

    documents = retrieve_documents(
        retriever=retriever,
        query=query,
    )

    # ========================================================
    # 6. Display results
    # ========================================================

    print(
        f"\nRetrieved {len(documents)} documents "
        f"(k={RETRIEVAL_K})"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print("\n" + "-" * 80)
        print(f"RESULT #{index}")
        print("-" * 80)

        print("\nMetadata:")

        print(
            f"  File: "
            f"{document.metadata.get('file_name', 'N/A')}"
        )

        print(
            f"  Page: "
            f"{document.metadata.get('page', 'N/A')}"
        )

        print(
            f"  Source: "
            f"{document.metadata.get('source', 'N/A')}"
        )

        print("\nContent:")
        print(document.page_content)


if __name__ == "__main__":
    main()