import json
import logging

from src.embeddings.embedding_model import create_embedding_model
from src.vectorstore.chroma_store import load_vectorstore
from src.vectorstore.retriever import create_retriever

from src.evaluation.safety_evaluation import run_task4_evaluation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():

    print("=" * 80)
    print("GlucoDoc Safety Evaluation")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Embedding Model
    # --------------------------------------------------------

    print("\n[1/3] Loading embedding model...")

    embedding_model = create_embedding_model()

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # 2. ChromaDB
    # --------------------------------------------------------

    print("\n[2/3] Loading ChromaDB...")

    vectorstore = load_vectorstore(
        embedding_model=embedding_model,
    )

    print("ChromaDB loaded.")

    # --------------------------------------------------------
    # 3. Retriever
    # --------------------------------------------------------

    print("\n[3/3] Creating retriever...")

    retriever = create_retriever(
        vectorstore=vectorstore,
    )

    print("Retriever ready.")

    # --------------------------------------------------------
    # Safety Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Running Safety Evaluation")
    print("=" * 80)

    results = run_task4_evaluation(
        retriever=retriever,
        k=4,
        embedding_model=embedding_model,
    )

    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CALIBRATION")
    print("=" * 80)

    print(
        json.dumps(
            results["calibration"],
            indent=4,
        )
    )

    print("\n")
    print("=" * 80)
    print("CLAIM DETECTOR")
    print("=" * 80)

    print(
        json.dumps(
            results["claim_detector"],
            indent=4,
        )
    )

    print("\n")
    print("=" * 80)
    print("BENCHMARK")
    print("=" * 80)

    print(
        json.dumps(
            results["benchmark"],
            indent=4,
        )
    )

    print("\n")
    print("=" * 80)
    print("SAFETY EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()