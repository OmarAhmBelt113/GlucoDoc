import logging

from src.embeddings.embedding_model import create_embedding_model
from src.rag.generator import create_llm
from src.rag.pipeline import run_rag
from src.vectorstore.chroma_store import load_vectorstore
from src.vectorstore.retriever import create_retriever


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main():

    print("=" * 80)
    print("GlucoDoc End-to-End RAG Test")
    print("=" * 80)

    # ========================================================
    # 1. Load Embedding Model
    # ========================================================

    print("\n[1/4] Loading embedding model...")

    embedding_model = create_embedding_model()

    print("Embedding model loaded.")

    # ========================================================
    # 2. Load ChromaDB
    # ========================================================

    print("\n[2/4] Loading ChromaDB...")

    vectorstore = load_vectorstore(
        embedding_model=embedding_model
    )

    print("ChromaDB loaded.")

    # ========================================================
    # 3. Create Retriever + LLM
    # ========================================================

    print("\n[3/4] Creating retriever and LLM...")

    retriever = create_retriever(
        vectorstore=vectorstore
    )

    llm = create_llm()

    print("Retriever and LLM ready.")

    # ========================================================
    # 4. Run RAG
    # ========================================================

    print("\n[4/4] Running RAG...")

    question = (
        "What are the diagnostic criteria for "
        "diabetes mellitus?"
    )

    answer = run_rag(
        question=question,
        retriever=retriever,
        llm=llm,
    )

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("RAG TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()