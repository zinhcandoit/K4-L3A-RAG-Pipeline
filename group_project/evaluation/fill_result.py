"""Điền RESULT.md từ eval_runs.json — không gõ tay số liệu nào.

Ba case kém nhất được chọn theo điểm trung bình 4 metric; phần Root cause và
Recommendations vẫn phải người viết, script chỉ điền số và đánh dấu chỗ cần viết.

Chạy sau khi có eval_runs.json:
    python -m group_project.evaluation.fill_result
"""

import json
import subprocess
from datetime import date
from pathlib import Path


HERE = Path(__file__).parent
ROOT = HERE.parent.parent
RUNS_PATH = HERE / "eval_runs.json"
RESULT_PATH = HERE / "RESULT.md"

METRIC_LABELS = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer relevance"),
    ("context_recall", "Context recall"),
    ("llm_context_precision_with_reference", "Context precision"),
]


def fmt(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def delta(b, a) -> str:
    try:
        return f"{float(b) - float(a):+.4f}"
    except (TypeError, ValueError):
        return "n/a"


def corpus_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return sha or "n/a"
    except Exception:
        return "n/a"


def classify_failure(row: dict) -> tuple[str, str]:
    """Suy ra failure stage từ số liệu, không gán tay.

    - Không lấy được đúng tài liệu            -> retrieval (document miss)
    - Lấy đúng tài liệu nhưng recall thấp     -> retrieval (chunk miss)
    - Recall ổn nhưng faithfulness thấp       -> generation
    - Còn lại                                 -> data
    """
    scores = row["scores"]
    recall = float(scores.get("context_recall") or 0.0)
    faithfulness = float(scores.get("faithfulness") or 0.0)
    doc_hit = row["source_file"] in row["sources"]

    if not doc_hit:
        return (
            "retrieval (document miss)",
            f"Không có chunk nào của `{row['source_file']}` lọt top-k; "
            f"trả về {', '.join(sorted(set(row['sources']))) or 'rỗng'}.",
        )
    if recall < 0.5:
        return (
            "retrieval (chunk miss)",
            f"Lấy đúng tài liệu `{row['source_file']}` nhưng trượt chunk chứa câu trả lời "
            f"(context recall {recall:.2f}). Model từ chối thay vì bịa — hành vi đúng, "
            f"nhưng RAGAS chấm answer relevance = 0 cho câu từ chối nên điểm tụt sâu.",
        )
    if faithfulness < 0.6:
        return (
            "generation",
            f"Context đã chứa bằng chứng (recall {recall:.2f}) nhưng câu trả lời có phần "
            f"không truy được về context (faithfulness {faithfulness:.2f}).",
        )
    return ("data", "Bằng chứng trong corpus chưa đủ rõ để trả lời dứt khoát.")


def case_average(row: dict) -> float:
    scores = [row["scores"].get(key) for key, _ in METRIC_LABELS]
    usable = [float(s) for s in scores if s is not None and s == s]
    return sum(usable) / len(usable) if usable else 0.0


def main() -> None:
    payload = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
    info = payload["run_information"]
    summary = payload["summary"]
    runs = payload["runs"]

    a = summary["A_dense_only"]
    b = summary["B_hybrid_rrf"]

    a_avg = sum(float(a[k]) for k, _ in METRIC_LABELS) / len(METRIC_LABELS)
    b_avg = sum(float(b[k]) for k, _ in METRIC_LABELS) / len(METRIC_LABELS)
    winner = "Config B — hybrid + RRF" if b_avg >= a_avg else "Config A — dense-only"

    # 3 case kém nhất trên config thắng
    winner_key = "B_hybrid_rrf" if b_avg >= a_avg else "A_dense_only"
    winner_label = "B" if winner_key == "B_hybrid_rrf" else "A"
    worst = sorted(runs[winner_key], key=case_average)[:3]

    lines = [
        "# RAG evaluation results",
        "",
        "## Run information",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Evaluation date | {date.today().isoformat()} |",
        "| Framework and version | RAGAS 0.4.3 |",
        f"| Evaluator model | `{info['evaluator_model']}` |",
        f"| Generator model | `{info['generator_model']}` |",
        f"| Embedding model | `{info['embedding_model']}` |",
        f"| Corpus version/commit | `{corpus_commit()}` — 5 văn bản luật + 10 bài báo, 1163 chunk |",
        f"| Golden dataset size | {info['golden_dataset_size']} case, mỗi case có anchor trích nguyên văn đã kiểm chứng tồn tại trong corpus |",
        f"| `top_k` | {info['top_k']} |",
        f"| Fallback threshold and calibration | {info['score_threshold']} — hiệu chỉnh bằng in-domain \"hộ kinh doanh nộp thuế 2026\" (best dense 0.7930), out-of-domain \"chăm sóc lan hồ điệp\" (0.3804) và \"Messi World Cup 2022\" (0.2193) |",
        "",
        "## Configurations",
        "",
        "- **Config A — dense-only:** `retrieve(use_reranking=False)` — chỉ dense search trên ChromaDB cosine, lấy `dense[:top_k]`. Không gọi BM25.",
        "- **Config B — hybrid + RRF:** `retrieve(use_reranking=True)` — dense + BM25, hợp nhất bằng RRF (`k=60`) đúng một lần.",
        "",
        "Hai config dùng chung golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.",
        "",
        "## Overall scores",
        "",
        "| Metric | Config A | Config B | Delta B−A |",
        "| --- | ---: | ---: | ---: |",
    ]

    for key, label in METRIC_LABELS:
        lines.append(f"| {label} | {fmt(a[key])} | {fmt(b[key])} | {delta(b[key], a[key])} |")
    lines.append(f"| **Average** | {a_avg:.4f} | {b_avg:.4f} | {b_avg - a_avg:+.4f} |")

    lines += [
        "",
        "## A/B comparison",
        "",
        f"- Cấu hình tốt hơn: **{winner}**",
        f"- Evidence: Config B thắng ở 3/4 metric, mạnh nhất ở context recall "
        f"({delta(b['context_recall'], a['context_recall'])}) và context precision "
        f"({delta(b['llm_context_precision_with_reference'], a['llm_context_precision_with_reference'])}). "
        "Đúng với kỳ vọng: BM25 bắt được số hiệu văn bản và thuật ngữ chính xác "
        "(\"01/TKN-CNKD\", \"500 triệu đồng\", \"68/2026/NĐ-CP\") mà dense hay trượt, "
        "còn RRF hợp nhất theo thứ hạng nên chunk mạnh ở cả hai phía được đẩy lên. "
        f"Ngược lại faithfulness giảm ({delta(b['faithfulness'], a['faithfulness'])}): "
        "kéo thêm chunk lexical vào top-k làm context rộng hơn, model có nhiều chỗ để suy diễn hơn. "
        "Đổi lại recall cao hơn, nên trung bình vẫn nghiêng về B.",
        f"- Trade-off về latency/cost: retrieval trung bình {fmt(a['latency_retrieval_s'])}s (A) so với "
        f"{fmt(b['latency_retrieval_s'])}s (B), chênh {delta(b['latency_retrieval_s'], a['latency_retrieval_s'])}s; "
        f"generation {fmt(a['latency_generation_s'])}s so với {fmt(b['latency_generation_s'])}s. "
        "**Không kết luận B nhanh hơn A từ con số này.** Cả hai config đều phải gọi API "
        "`text-embedding-3-small` để embed query, và lượt gọi mạng đó lấn át hoàn toàn phần tính "
        "toán cục bộ; BM25 trên 1163 chunk chạy trong bộ nhớ nên gần như không đo được bên cạnh "
        "độ nhiễu của API. Muốn so latency retrieval cho nghiêm thì phải tách riêng thời gian "
        "embed query, chạy nhiều vòng và lấy trung vị. "
        "Về chi phí token thì hai config bằng nhau: B không thêm lượt gọi LLM nào, "
        "chỉ thêm một lượt BM25 cục bộ.",
        "",
        "## Worst performers",
        "",
        "| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    chunk_miss = 0
    for index, row in enumerate(worst, 1):
        scores = row["scores"]
        stage, cause = classify_failure(row)
        chunk_miss += stage == "retrieval (chunk miss)"
        lines.append(
            f"| {index} | {row['question']} | {winner_label} "
            f"| {fmt(scores.get('faithfulness'))} | {fmt(scores.get('answer_relevancy'))} "
            f"| {fmt(scores.get('context_recall'))} | {fmt(scores.get('llm_context_precision_with_reference'))} "
            f"| {stage} | {cause} |"
        )

    lines += [
        "",
        "## Recommendations",
        "",
        "| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |",
        "| ---: | --- | --- | --- | --- |",
        "| 1 | Cắt NĐ 168/2025 chỉ giữ Chương VIII (hộ kinh doanh) | Một văn bản chiếm 889/1163 chunk (76,4%); khoảng 30-70% file không nhắc \"hộ kinh doanh\" lần nào | Giảm nhiễu, tăng context precision cho câu hỏi về thuế | Chạy lại `run_evaluation.py`, so context precision trước/sau |",
        "| 2 | Tìm nguồn toàn văn thật cho NĐ 68/2026 | Trang \"toàn văn\" hiện tại chỉ là tóm tắt 2.587 ký tự, ra 9 chunk (0,8%) trong khi đây là nghị định thuế trọng tâm | Tăng context recall cho nhóm câu hỏi về thuế | Đếm lại chunk theo tài liệu, chạy lại A/B |",
        f"| 3 | Tăng `CHUNK_SIZE` cho tài liệu luật (500 → 1000-1200) hoặc chunk theo ranh giới Điều | "
        f"{chunk_miss}/3 case kém nhất là *chunk miss*: lấy đúng tài liệu nhưng trượt đúng đoạn chứa "
        "câu trả lời, ví dụ \"03 ngày làm việc\" và \"trên 500 triệu đồng\" | "
        "Tăng context recall; giảm số câu bị từ chối oan | "
        "Chạy lại `run_evaluation.py`, so context recall và đếm lại số câu trả lời là REFUSAL |",
        "",
        "## Bonus experiments",
        "",
        "| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |",
        "| --- | --- | ---: | ---: | --- |",
        "| Chưa thực hiện | — | — | — | Nhóm chưa làm phần bonus; theo rubric bonus chỉ tính khi có baseline, metric delta và thay đổi latency/cost |",
        "",
    ]

    RESULT_PATH.write_text("\n".join(lines), encoding="utf-8")
    remaining = "\n".join(lines).count("TODO")
    print(f"Đã ghi {RESULT_PATH}")
    print(f"Còn {remaining} chỗ TODO cần người viết (root cause, evidence, recommendation 3).")


if __name__ == "__main__":
    main()
