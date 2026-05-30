from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.embeddings import GoogleEmbeddings
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from openai import AsyncOpenAI
import os
import instructor
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
import math
import time
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()


def load_rag_outputs(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evaluation_dataset(rag_outputs: list[dict]) -> EvaluationDataset:
    """
    Converts RAG outputs into RAGAS EvaluationDataset.
    The field names change slightly: question -> user_input, answer -> response, contexts -> retrieved_contexts.
    """

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
            reference=row["reference"]
        )

        for row in rag_outputs
    ]
    return EvaluationDataset(samples=samples)


def setup_evaluator():
    """
    Builds the RAGAS evaluator LLM and embeddings.
    AsyncOpenAI points at Gemini's OpenAI-compatible endpoint.
    Mode.MD_JSON avoids function calling, which causes Gemini to return
    empty lists for RAGAS's nested schemas - producing NaN faithfulness scores.
    """

    raw_client = AsyncOpenAI(
        api_key=os.environ["GOOGLE_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    patched_client = instructor.from_openai(
        raw_client, mode=instructor.Mode.MD_JSON)

    llm = InstructorLLM(
        client=patched_client,
        model="gemini-2.5-flash",
        provider="openai",
        model_args=InstructorModelArgs(max_tokens=4096)
    )

    # RAGAS native google embeddings class
    embeddings = GoogleEmbeddings(model="gemini-embedding-001")

    return llm, embeddings


async def _score_all(samples: list[SingleTurnSample], llm, embbedings) -> dict[str, float]:
    """
    Calls ascore() on each sample for all four metrics.
    All four metrics run concurrently per sample via asyncio.gather
    The avg() function skips any NaN values so one failed sample doesn't posion the entire matic score.
    """

    # Instantiate the four metrics with the evaluator LLM
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embbedings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    f_scores, ar_scores, cp_scores, cr_scores = [], [], [], []

    for i, sample in enumerate(samples):
        print(f"Scoring sample {i+1}/{len(samples)}....")

        # Verify all required fields are present
        assert sample.user_input is not None
        assert sample.response is not None
        assert sample.retrieved_contexts is not None
        assert sample.reference is not None

        # Score each metric sequentially with delays to stay within
        f = await faithfulness.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )
        await asyncio.sleep(20)

        ar = await answer_relevancy.ascore(
            user_input=sample.user_input,
            response=sample.response,
        )
        await asyncio.sleep(20)

        cp = await context_precision.ascore(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )
        await asyncio.sleep(20)

        cr = await context_recall.ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            reference=sample.reference,
        )
        await asyncio.sleep(20)  # wait before next sample

        f_scores.append(f)
        ar_scores.append(ar)
        cp_scores.append(cp)
        cr_scores.append(cr)

    def avg(lst: list[float]) -> float:
        # Filter out NaN before averaging one bad sample shouldn't sink the metric
        valid = [x for x in lst if not math.isnan(x)]
        return round(sum(valid) / len(valid), 4) if valid else float("nan")

    return {
        "faithfulness": avg(f_scores),
        "answer_relevancy": avg(ar_scores),
        "context_precision": avg(cp_scores),
        "context_recall": avg(cr_scores)
    }


def run_evaluation(dataset: EvaluationDataset, llm, embeddings) -> dict[str, float]:
    print("Running RAGAS evaluation (1-3 minutes)....")

    # Filter to SingleTurnSample only - EvaluationDataset can hold mixed types
    samples = [s for s in dataset.samples if isinstance(s, SingleTurnSample)]
    return asyncio.run(_score_all(samples, llm, embeddings))


if __name__ == "__main__":
    print("Loading RAG outputs...")
    rag_outputs = load_rag_outputs("rag_outputs.json")
    print(f"Loaded {len(rag_outputs)} samples. \n")

    print("Building evaluation dataset...")
    dataset = build_evaluation_dataset(rag_outputs)
    print("Dataset ready. \n")

    print("Setting up evaluator (Gemini 2.5. Flash)")
    llm, embeddings = setup_evaluator()
    print("Evaluator ready. \n")

    scores = run_evaluation(dataset, llm, embeddings)

    print("\n" + "=" * 60)
    print("RAGAS Evaluation Results")
    print("=" * 60)
    for name, score in scores.items():
        print(f" {name}: {score:.4f}")

    with open("eval_results.json", "w", encoding="utf-8") as f:

        json.dump(scores, f, indent=2)
    print("Results saved to eval_results.json")
