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

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embbedings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    f_scores, ar_scores, cp_scores, cr_scores = [], [], [], []

    for i, sample in enumerate(samples):
        print(f"Scoring sample {i+1}/{len(samples)}....")

        assert sample.user_input is not None
        assert sample.response is not None
        assert sample.retrieved_contexts is not None
        assert sample.reference is not None
