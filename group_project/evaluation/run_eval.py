"""
Đánh giá RAG: Config A (dense-only) vs Config B (hybrid + RRF).

4 metric theo định nghĩa của RAGAS, cài đặt bằng LLM-as-judge (Gemini):
  - faithfulness       : tỉ lệ claim trong câu trả lời được context hỗ trợ
  - answer_relevancy   : cosine trung bình giữa câu hỏi gốc và 3 câu hỏi sinh ngược từ câu trả lời
  - context_recall     : tỉ lệ câu trong ground_truth có thể suy ra từ context
  - context_precision  : average precision của các chunk theo thứ hạng truy xuất

Chạy: python -m group_project.evaluation.run_eval [--limit N]
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.task4_chunking_indexing import embed_texts  # noqa: E402
from src.task9_retrieval_pipeline import retrieve  # noqa: E402
from src.task10_generation import (  # noqa: E402
    LLM_MODEL,
    SYSTEM_PROMPT,
    call_llm,
    format_context,
    reorder_for_llm,
)

HERE = Path(__file__).parent
GOLDEN = HERE / "golden_dataset.json"
OUT = HERE / "eval_results.json"
TOP_K = 5
WORKERS = 1
CONFIGS = {"A": False, "B": True}  # config -> use_reranking


MIN_INTERVAL = 13.0  # free tier: 5 request/phút/model
_LOCK = threading.Lock()
_LAST = [0.0]


def throttle() -> None:
    with _LOCK:
        wait = _LAST[0] + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()


def limited_call_llm(system_prompt: str, user_message: str) -> str:
    last = None
    for attempt in range(6):
        throttle()
        try:
            return call_llm(system_prompt, user_message)
        except Exception as error:  # noqa: BLE001
            last = error
            time.sleep(retry_delay(error))
    raise RuntimeError(f"generation failed: {last}")


def retry_delay(error: Exception) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(error))
    return float(match.group(1)) + 2 if match else 10.0


def judge(prompt: str) -> dict:
    """Gọi Gemini ở chế độ JSON, retry khi lỗi tạm thời."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    last = None
    for attempt in range(6):
        throttle()
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0, response_mime_type="application/json"
                ),
            )
            text = (response.text or "").strip()
            text = re.sub(r"^```(?:json)?|```$", "", text).strip()
            return json.loads(text)
        except Exception as error:  # noqa: BLE001
            last = error
            time.sleep(retry_delay(error))
    raise RuntimeError(f"judge failed: {last}")


def ctx_text(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {c['content']}" for i, c in enumerate(chunks, 1))


def judge_all(question: str, ground_truth: str, answer: str, chunks: list[dict]) -> dict:
    """Một lần gọi judge cho cả 4 metric (tiết kiệm quota)."""
    n = len(chunks)
    prompt = (
        "Bạn là giám khảo đánh giá hệ thống RAG. Thực hiện 4 việc sau, chỉ dựa trên nội dung được cung cấp.\n\n"
        f"CÂU HỎI: {question}\n\nĐÁP ÁN CHUẨN: {ground_truth}\n\n"
        f"CÂU TRẢ LỜI CỦA HỆ THỐNG:\n{answer}\n\n"
        f"CONTEXT (các chunk truy xuất, theo thứ hạng):\n{ctx_text(chunks)}\n\n"
        "1) claims: tách câu trả lời thành các khẳng định độc lập; với mỗi claim, supported=true nếu có thể suy ra "
        "trực tiếp từ CONTEXT. Nếu câu trả lời chỉ là lời từ chối thì claims rỗng.\n"
        "2) questions: viết 3 câu hỏi khác nhau mà CÂU TRẢ LỜI có thể đang trả lời (rỗng nếu là lời từ chối).\n"
        "3) statements: tách ĐÁP ÁN CHUẨN thành các ý riêng; attributable=true nếu ý đó suy ra được từ CONTEXT.\n"
        f"4) relevant: danh sách đúng {n} giá trị true/false, theo thứ tự chunk, "
        "true nếu chunk hữu ích để đưa ra ĐÁP ÁN CHUẨN.\n\n"
        'Trả về JSON: {"claims": [{"claim": str, "supported": bool}], "questions": [str], '
        '"statements": [{"statement": str, "attributable": bool}], "relevant": [bool]}'
    )
    result = judge(prompt)
    claims = result.get("claims") or []
    faith = sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else 0.0
    statements = result.get("statements") or []
    recall = (
        sum(bool(x.get("attributable")) for x in statements) / len(statements)
        if statements else 0.0
    )
    flags = [bool(v) for v in (result.get("relevant") or [])][: len(chunks)]
    flags += [False] * (len(chunks) - len(flags))
    hits, total = 0, 0.0
    for k, flag in enumerate(flags, 1):
        if flag:
            hits += 1
            total += hits / k
    precision = total / hits if hits else 0.0
    questions = [q for q in (result.get("questions") or []) if isinstance(q, str) and q.strip()]
    return {
        "faithfulness": faith,
        "context_recall": recall,
        "context_precision": precision,
        "gen_questions": questions,
    }


def run_item(item: dict, config: str, chunks: list[dict]) -> dict:
    question = item["question"]
    started = time.time()
    if chunks:
        context = format_context(reorder_for_llm(chunks))
        try:
            answer = limited_call_llm(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {question}")
        except Exception as error:  # noqa: BLE001
            answer = ""
            print(f"  ! generation error {item['id']}/{config}: {error}")
    else:
        answer = ""
    latency = time.time() - started
    if not answer.strip():
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return {
        "id": item["id"],
        "config": config,
        "question": question,
        "answer": answer,
        "retrieved_ids": [c["id"] for c in chunks],
        "retrieval_methods": sorted({c["retrieval_method"] for c in chunks}),
        "gen_latency_s": round(latency, 1),
        **judge_all(question, item["ground_truth"], answer, chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = json.loads(GOLDEN.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        items = items[: args.limit]

    # Retrieval chạy tuần tự (dùng chung model embedding), sau đó gọi LLM song song.
    jobs, retrieval_time = [], {"A": 0.0, "B": 0.0}
    for item in items:
        for config, rerank in CONFIGS.items():
            t = time.time()
            chunks = retrieve(item["question"], top_k=TOP_K, use_reranking=rerank)
            retrieval_time[config] += time.time() - t
            jobs.append((item, config, chunks))
    print(f"retrieval done: {len(jobs)} jobs")

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(run_item, *job) for job in jobs]
        rows = []
        for done, future in enumerate(futures, 1):
            rows.append(future.result())
            print(f"  {done}/{len(futures)} done")

    # answer_relevancy: cosine(question, câu hỏi sinh ngược) bằng embedding dùng chung
    for row in rows:
        qs = row.pop("gen_questions")
        if not qs:
            row["answer_relevancy"] = 0.0
            continue
        vectors = np.array(embed_texts([row["question"]] + qs))
        row["answer_relevancy"] = float(np.mean(vectors[1:] @ vectors[0]))

    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    summary = {}
    for config in CONFIGS:
        sel = [r for r in rows if r["config"] == config]
        summary[config] = {m: float(np.mean([r[m] for r in sel])) for m in metrics}
        summary[config]["average"] = float(np.mean([summary[config][m] for m in metrics]))
        summary[config]["avg_retrieval_latency_s"] = retrieval_time[config] / len(items)
        summary[config]["avg_generation_latency_s"] = float(np.mean([r["gen_latency_s"] for r in sel]))

    OUT.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
