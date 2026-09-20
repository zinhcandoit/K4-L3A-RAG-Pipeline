"""Chạy A/B dense-only vs hybrid+RRF trên golden dataset và chấm bằng RAGAS.

Hai config chỉ khác retrieval strategy; golden dataset, generator, prompt,
top_k và evaluator đều giữ nguyên.

    Config A — dense-only    : retrieve(use_reranking=False) -> dense[:top_k]
    Config B — hybrid + RRF  : retrieve(use_reranking=True)

Chạy:
    python -m group_project.evaluation.run_evaluation --limit 2   # thử trước cho rẻ
    python -m group_project.evaluation.run_evaluation             # chạy đủ
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).parent.parent.parent
load_dotenv(ROOT / ".env")

from src.task9_retrieval_pipeline import SCORE_THRESHOLD, retrieve  # noqa: E402
from src.task10_generation import (  # noqa: E402
    REFUSAL,
    SYSTEM_PROMPT,
    call_llm,
    format_context,
    reorder_for_llm,
)

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
RUNS_PATH = Path(__file__).parent / "eval_runs.json"

TOP_K = 5
GENERATOR_MODEL = os.getenv("LLM_MODEL", "")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")

CONFIGS = {
    "A_dense_only": {"use_reranking": False, "label": "Config A — dense-only"},
    "B_hybrid_rrf": {"use_reranking": True, "label": "Config B — hybrid + RRF"},
}

METRIC_ORDER = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "llm_context_precision_with_reference",
]


def answer_one(question: str, use_reranking: bool) -> dict:
    """Chạy retrieve + generate cho một câu, đo latency từng pha."""
    started = time.perf_counter()
    chunks = retrieve(question, top_k=TOP_K, use_reranking=use_reranking)
    retrieval_seconds = time.perf_counter() - started

    if not chunks:
        return {
            "answer": REFUSAL,
            "contexts": [],
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": 0.0,
            "sources": [],
        }

    context = format_context(reorder_for_llm(chunks))
    started = time.perf_counter()
    try:
        answer = call_llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
    except Exception as error:
        print(f"    [generation lỗi] {error}")
        answer = REFUSAL
    generation_seconds = time.perf_counter() - started

    return {
        "answer": answer or REFUSAL,
        "contexts": [chunk["content"] for chunk in chunks],
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "sources": [chunk["metadata"]["source"] for chunk in chunks],
    }


def warm_up() -> None:
    """Nạp corpus và dựng index BM25 trước khi đo.

    Lần gọi đầu tiên phải trả chi phí dựng index một lần; nếu không warm-up,
    config nào chạy trước sẽ gánh hết và số latency A/B thành vô nghĩa.
    """
    print("Warm-up (nạp corpus + dựng index BM25)...")
    started = time.perf_counter()
    retrieve("khởi động", top_k=1, use_reranking=True)
    print(f"  xong sau {time.perf_counter() - started:.2f}s\n")


def collect_runs(cases: list[dict]) -> dict:
    warm_up()
    runs = {}
    for key, config in CONFIGS.items():
        print(f"\n=== {config['label']} ===")
        rows = []
        for index, case in enumerate(cases, 1):
            print(f"  [{index}/{len(cases)}] {case['question'][:60]}...")
            output = answer_one(case["question"], config["use_reranking"])
            rows.append({**case, **output})
        runs[key] = rows
    return runs


def score_with_ragas(rows: list[dict]) -> dict:
    """Chấm 4 metric RAGAS cho một config."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"] or [""],
            reference=row["expected_answer"],
        )
        for row in rows
    ]

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=EVALUATOR_MODEL))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
    )
    return result.to_pandas().to_dict(orient="list")


def mean(values) -> float:
    numbers = [float(v) for v in values if v is not None and str(v) != "nan" and v == v]
    return statistics.fmean(numbers) if numbers else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="chỉ chạy N case đầu")
    args = parser.parse_args()

    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    print(f"Golden dataset : {len(cases)} case")
    print(f"Generator      : {GENERATOR_MODEL}")
    print(f"Evaluator      : {EVALUATOR_MODEL}")
    print(f"Embedding      : {EMBEDDING_MODEL}")
    print(f"top_k          : {TOP_K} · threshold {SCORE_THRESHOLD}")

    runs = collect_runs(cases)

    summary = {}
    for key, rows in runs.items():
        print(f"\n=== Chấm RAGAS: {CONFIGS[key]['label']} ===")
        scored = score_with_ragas(rows)
        for index, row in enumerate(rows):
            row["scores"] = {
                metric: scored[metric][index] for metric in METRIC_ORDER if metric in scored
            }
        summary[key] = {
            metric: mean(scored.get(metric, [])) for metric in METRIC_ORDER
        }
        summary[key]["latency_retrieval_s"] = mean(r["retrieval_seconds"] for r in rows)
        summary[key]["latency_generation_s"] = mean(r["generation_seconds"] for r in rows)

    payload = {
        "run_information": {
            "generator_model": GENERATOR_MODEL,
            "evaluator_model": EVALUATOR_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
            "golden_dataset_size": len(cases),
        },
        "summary": summary,
        "runs": runs,
    }
    RUNS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'Metric':<40} {'A dense':>10} {'B hybrid':>10} {'Delta':>10}")
    print("-" * 74)
    for metric in METRIC_ORDER + ["latency_retrieval_s", "latency_generation_s"]:
        a = summary["A_dense_only"].get(metric, float("nan"))
        b = summary["B_hybrid_rrf"].get(metric, float("nan"))
        print(f"{metric:<40} {a:>10.4f} {b:>10.4f} {b - a:>+10.4f}")
    print(f"\nChi tiết từng case -> {RUNS_PATH}")


if __name__ == "__main__":
    main()
