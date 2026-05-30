from rag_pipeline import build_vector_stor, rag_answer
import json
from dotenv import load_dotenv
load_dotenv()

TEST_QUESTION = [
    {
        "question": "What is RAG and why does it reduce hallucinations?",
        "reference": (
            "RAG stands for Retrieval-Augmented Generation. It retrieves relevant "
            "documents from an external database before generating an answer, "
            "grounding the response in real content. This reduces hallucinations "
            "because the LLM answers from retrieved documentsrather than from memory."
        ),
    },
    {
        "question": "What does Faithfulness measure in RAGAS?",
        "reference": (
            "Faithfulness measures whether every claim in the generated answer can be "
            "traced back to the retrieved context. An unfaithful answer contains "
            "hallucinations - claims not supported by the retrieved documents."
        )
    },
    {
        "question": "How is AnswerRelevancy different from Faithfulness?",
        "reference": (
            "Faithfulness checks whether the answer is grounded in the retrieved "
            "context. AnswerRelevancy checks whether the answer actually addresses "
            "the user's question. An answer can be faithful but still score low on "
            "relevancy if it answers the wrong question."
        ),
    },
    {
        "question": "How are RAGAS v0.4 collections metrics scored?",
        "reference": (
            "RAGAS v0.4 collections metrics are scored by calling ascore() directly "
            "on each sample. The evaluator LLM is initialized using InstructorLLM "
            "with Mode.MD_JSON and an AsyncOpenAI client pointed at Gemini's "
            "OpenAI-compatible endpoint."
        ),
    },
    {
        "question": "What embedding model is used in the RAG pipeline?",
        "reference": (
            "The all-MiniLM-L6-v2 model from HuggingFace is used. It runs entirely "
            "locally with no API key needed."
        ),
    },
]


def load_corpus(filepath: str) -> list[str]:
    """
    Read the corpus file and splits it into non-empty paragraphs.
    """

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def collect_rag_outputs() -> list[dict]:
    """
    Runs the RAG pipeline on all test questions.
    Returns a list of dicts - one per questions - containing the question,
    the generated answer, the retrieved context chunks, and the reference answer
    """

    print("Building vector store...")
    corpus = load_corpus("data/docs.txt")
    vector_store = build_vector_stor(corpus)
    print(f"Indexed {len(corpus)} documents. \n")

    result = []
    for i, item in enumerate(TEST_QUESTION):
        question = item["question"]
        print(f"[{i+1}/{len(TEST_QUESTION)}] {question[:65]}....")

        answer, contexts = rag_answer(question, vector_store)
        result.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference": item["reference"]
        })
        print(f"-> {answer[:80]}....\n")

    return result


if __name__ == "__main__":
    result = collect_rag_outputs()
    with open("rag_outputs.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(result)} results to rag_outputs.json")
