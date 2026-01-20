from api.agents.retrieval_generation import rag_pipeline

import asyncio

from qdrant_client import QdrantClient

from langsmith import Client
from qdrant_client import QdrantClient

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy

ls_client = Client()
qdrant_client = QdrantClient(
    url=f"http://localhost:6333"
)

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

def _get_question(run, example):
    run_inputs = run.inputs or {}
    if "question" in run_inputs:
        return run_inputs["question"]
    example_inputs = example.inputs or {}
    return example_inputs.get("question")


def _get_outputs(run):
    return run.outputs or {}


async def ragas_faithfulness(run, example, **_):
    run_inputs = run.inputs or {}
    run_outputs = _get_outputs(run)
    question = run_inputs.get("question") or _get_question(run, example)

    sample = SingleTurnSample(
            user_input=question,
            response=run_outputs.get("answer"),
            retrieved_contexts=run_outputs.get("retrieved_context")
        )
    scorer = Faithfulness(llm=ragas_llm)

    return await scorer.single_turn_ascore(sample)


async def ragas_responce_relevancy(run, example, **_):
    run_outputs = _get_outputs(run)
    question = _get_question(run, example)

    sample = SingleTurnSample(
            user_input=question,
            response=run_outputs.get("answer"),
            retrieved_contexts=run_outputs.get("retrieved_context")
        )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    return await scorer.single_turn_ascore(sample)


async def ragas_context_precision_id_based(run, example, **_):
    run_outputs = _get_outputs(run)
    example_outputs = example.outputs or {}

    sample = SingleTurnSample(
            retrieved_context_ids=run_outputs.get("retrieved_context_ids"),
            reference_context_ids=example_outputs.get("reference_context_ids")
        )
    scorer = IDBasedContextPrecision()

    return await scorer.single_turn_ascore(sample)


async def ragas_context_recall_id_based(run, example, **_):
    run_outputs = _get_outputs(run)
    example_outputs = example.outputs or {}

    sample = SingleTurnSample(
            retrieved_context_ids=run_outputs.get("retrieved_context_ids"),
            reference_context_ids=example_outputs.get("reference_context_ids")
        )
    scorer = IDBasedContextRecall()

    return await scorer.single_turn_ascore(sample)


async def run_rag_pipeline(example):
    return await asyncio.to_thread(rag_pipeline, example["question"], qdrant_client)


results = asyncio.run(ls_client.aevaluate(
    run_rag_pipeline,
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_responce_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
     ],
     experiment_prefix="retriever",
     max_concurrency=10
))