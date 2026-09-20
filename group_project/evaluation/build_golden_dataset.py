"""Dựng golden dataset và kiểm chứng từng case có thật trong corpus.

Mỗi case khai một `anchor` — chuỗi trích nguyên văn từ tài liệu. Script bắt buộc
anchor phải xuất hiện trong data/standardized/, nếu không thì báo lỗi và không ghi
file. Làm vậy để không có case nào được viết từ trí nhớ hoặc từ suy đoán.

Chạy:
    python -m group_project.evaluation.build_golden_dataset
"""

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
OUTPUT = Path(__file__).parent / "golden_dataset.json"

# question / expected_answer / anchor (trích nguyên văn) / source
CASES = [
    {
        "question": "Hồ sơ đăng ký thành lập hộ kinh doanh gồm những giấy tờ gì?",
        "expected_answer": "Hồ sơ đăng ký hộ kinh doanh bao gồm Giấy đề nghị đăng ký hộ kinh doanh, và trong trường hợp nhóm cá nhân tham gia hộ kinh doanh thì có thêm biên bản họp nhóm cá nhân về việc thành lập hộ kinh doanh.",
        "anchor": "Giấy đề nghị đăng ký hộ kinh doanh",
        "source": "nd-168-2025-dang-ky-doanh-nghiep-ho-kinh-doanh.md",
    },
    {
        "question": "Nộp hồ sơ đăng ký thành lập hộ kinh doanh ở cơ quan nào?",
        "expected_answer": "Chủ hộ kinh doanh hoặc người được ủy quyền gửi hồ sơ đến Cơ quan đăng ký kinh doanh cấp xã nơi hộ kinh doanh đăng ký trụ sở.",
        "anchor": "Cơ quan đăng ký kinh doanh cấp xã nơi hộ kinh doanh đăng ký trụ sở",
        "source": "nd-168-2025-dang-ky-doanh-nghiep-ho-kinh-doanh.md",
    },
    {
        "question": "Tên hộ kinh doanh có được trùng với hộ khác không?",
        "expected_answer": "Tên tiếng Việt của hộ kinh doanh không được trùng với tên tiếng Việt của hộ kinh doanh đã đăng ký trong phạm vi cấp xã, trừ những hộ đã chấm dứt hoạt động. Tên trùng là tên viết hoàn toàn giống nhau, không kể chữ hoa hay chữ thường.",
        "anchor": "không được trùng với tên tiếng Việt của hộ kinh doanh đã đăng ký trong phạm vi cấp xã",
        "source": "nd-168-2025-dang-ky-doanh-nghiep-ho-kinh-doanh.md",
    },
    {
        "question": "Một hộ kinh doanh được mở bao nhiêu địa điểm kinh doanh?",
        "expected_answer": "Một hộ kinh doanh có thể có nhiều địa điểm kinh doanh trong phạm vi cả nước. Địa điểm kinh doanh là nơi hộ tiến hành hoạt động kinh doanh cụ thể ngoài trụ sở, và hộ phải thông báo cho cơ quan quản lý thuế cùng cơ quan quản lý thị trường nơi đặt địa điểm.",
        "anchor": "Một hộ kinh doanh có thể có nhiều địa điểm kinh doanh trong phạm vi cả nước",
        "source": "nd-168-2025-dang-ky-doanh-nghiep-ho-kinh-doanh.md",
    },
    {
        "question": "Khi Giấy chứng nhận đăng ký hộ kinh doanh ghi sai so với hồ sơ thì bao lâu được cấp lại?",
        "expected_answer": "Cơ quan đăng ký kinh doanh cấp xã gửi thông báo hiệu đính và cấp lại Giấy chứng nhận đăng ký hộ kinh doanh trong thời hạn 03 ngày làm việc kể từ ngày gửi thông báo.",
        "anchor": "trong thời hạn 03 ngày làm việc",
        "source": "nd-168-2025-dang-ky-doanh-nghiep-ho-kinh-doanh.md",
    },
    {
        "question": "Nghị định 68/2026/NĐ-CP có bao nhiêu chương và bao nhiêu điều?",
        "expected_answer": "Nghị định 68/2026/NĐ-CP quy định về chính sách thuế và quản lý thuế đối với hộ kinh doanh, cá nhân kinh doanh gồm 5 Chương, 19 Điều.",
        "anchor": "gồm 5 Chương, 19 Điều",
        "source": "nd-68-2026-chinh-sach-thue-ho-kinh-doanh.md",
    },
    {
        "question": "Ngưỡng doanh thu năm nào được Nghị định 68/2026 dùng để quy định riêng việc khai thuế?",
        "expected_answer": "Nghị định 68/2026/NĐ-CP có quy định riêng về khai thuế, nộp thuế giá trị gia tăng và thuế thu nhập cá nhân đối với hộ kinh doanh, cá nhân kinh doanh có doanh thu năm trên 500 triệu đồng.",
        "anchor": "trên 500 triệu đồng",
        "source": "nd-68-2026-chinh-sach-thue-ho-kinh-doanh.md",
    },
    {
        "question": "Hộ kinh doanh không chịu thuế giá trị gia tăng thì thông báo doanh thu theo mẫu nào?",
        "expected_answer": "Theo Điều 4 Thông tư 18/2026/TT-BTC, hộ kinh doanh, cá nhân kinh doanh thuộc đối tượng không chịu thuế giá trị gia tăng và không phải nộp thuế thu nhập cá nhân thông báo doanh thu thực tế phát sinh trong năm và kê khai các loại thuế khác theo Mẫu số 01/TKN-CNKD.",
        "anchor": "Mẫu số 01/TKN-CNKD",
        "source": "tt-18-2026-btc-thu-tuc-thue-ho-kinh-doanh.md",
    },
    {
        "question": "Biểu mẫu dùng trong đăng ký hộ kinh doanh nằm ở phụ lục nào của Thông tư 68/2025/TT-BTC?",
        "expected_answer": "Biểu mẫu sử dụng trong đăng ký hộ kinh doanh được quy định tại Phụ lục 2 ban hành kèm theo Thông tư 68/2025/TT-BTC.",
        "anchor": "Biểu mẫu sử dụng trong đăng ký hộ kinh doanh",
        "source": "tt-68-2025-btc-bieu-mau-dang-ky-ho-kinh-doanh.md",
    },
    {
        "question": "Hộ kinh doanh vay vốn làm dự án xanh được hỗ trợ lãi suất bao nhiêu?",
        "expected_answer": "Theo Điều 9 Nghị quyết 198/2025/QH15, doanh nghiệp thuộc khu vực kinh tế tư nhân, hộ kinh doanh và cá nhân kinh doanh được Nhà nước hỗ trợ lãi suất 2%/năm khi vay vốn để thực hiện các dự án xanh, tuần hoàn và áp dụng khung tiêu chuẩn ESG.",
        "anchor": "hỗ trợ lãi suất 2%/năm khi vay vốn để thực hiện các dự án xanh",
        "source": "nq-198-2025-qh15-kinh-te-tu-nhan.md",
    },
    {
        "question": "Hộ kinh doanh tuân thủ tốt pháp luật thì được ưu đãi gì về kiểm tra?",
        "expected_answer": "Nghị quyết 198/2025/QH15 quy định miễn kiểm tra thực tế tại doanh nghiệp, hộ kinh doanh, cá nhân kinh doanh đối với những đối tượng tuân thủ tốt quy định của pháp luật.",
        "anchor": "Miễn kiểm tra thực tế tại doanh nghiệp, hộ kinh doanh, cá nhân kinh doanh",
        "source": "nq-198-2025-qh15-kinh-te-tu-nhan.md",
    },
    {
        "question": "Lệ phí môn bài với hộ kinh doanh chấm dứt từ khi nào?",
        "expected_answer": "Lệ phí môn bài với hộ kinh doanh chấm dứt từ ngày 1/1/2026, theo Nghị quyết 198 của Quốc hội về hỗ trợ thuế phí cho kinh tế tư nhân.",
        "anchor": "lệ phí môn bài với hộ kinh doanh sẽ chấm dứt từ 1/1/2026",
        "source": "bo-thue-khoan-ho-kinh-doanh-tinh-thue-the-nao-tu-2026-4992081.md",
    },
    {
        "question": "Khác biệt cơ bản giữa hộ nộp thuế khoán và hộ kê khai là gì?",
        "expected_answer": "Cách tính thuế về cơ bản giống nhau; điểm khác biệt duy nhất nằm ở doanh thu: hộ khoán dùng doanh thu ước tính ổn định từ đầu năm, còn hộ kê khai dùng số thực tế phát sinh.",
        "anchor": "hộ khoán sử dụng doanh thu ước tính ổn định từ đầu năm",
        "source": "bo-thue-khoan-ho-kinh-doanh-tinh-thue-the-nao-tu-2026-4992081.md",
    },
    {
        "question": "Hộ kinh doanh chuyển từ khoán sang kê khai có phải trả phí hay mua phần mềm không?",
        "expected_answer": "Không. Mọi thủ tục cho hộ kinh doanh chuyển từ hình thức khoán sang kê khai đều hoàn toàn miễn phí, không thu thêm lệ phí và không bắt buộc hộ phải mua phần mềm.",
        "anchor": "hoàn toàn miễn phí",
        "source": "ho-kinh-doanh-chuyen-sang-ke-khai-khong-can-mua-phan-mem-nop-le-phi-4958714.md",
    },
    {
        "question": "Hộ đang dùng hóa đơn điện tử thông thường có phải chuyển sang hóa đơn từ máy tính tiền không?",
        "expected_answer": "Có. Nếu chưa khởi tạo hóa đơn từ máy tính tiền thì hộ kinh doanh cần chuyển sang theo đúng quy định.",
        "anchor": "nếu chưa khởi tạo hóa đơn từ máy tính tiền thì cần chuyển sang theo đúng quy định",
        "source": "tu-khoan-sang-ke-khai-ho-kinh-doanh-hoi-co-quan-thue-tra-loi-102250612170705652.md",
    },
    {
        "question": "Chế độ kế toán cho hộ kinh doanh được điều chỉnh theo hướng nào khi bỏ thuế khoán?",
        "expected_answer": "Chế độ kế toán cho doanh nghiệp siêu nhỏ và hộ kinh doanh được điều chỉnh theo hướng đơn giản, không phát sinh thêm nhân sự kế toán chuyên trách.",
        "anchor": "chế độ kế toán cho doanh nghiệp siêu nhỏ và hộ kinh doanh cũng được điều chỉnh theo hướng đơn giản",
        "source": "xoa-bo-thue-khoan-chuyen-sang-ke-khai-nganh-thue-cam-tay-chi-viec-ho-tro-ho-kinh.md",
    },
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def load_corpus() -> dict[str, str]:
    return {
        path.name: normalize(path.read_text(encoding="utf-8"))
        for path in STANDARDIZED_DIR.rglob("*.md")
    }


def build() -> list[dict]:
    corpus = load_corpus()
    whole = " ".join(corpus.values())
    dataset, problems = [], []

    for index, case in enumerate(CASES, 1):
        anchor = normalize(case["anchor"])

        if anchor not in whole:
            problems.append(f"case {index}: anchor KHÔNG có trong corpus — {case['anchor'][:60]!r}")
            continue
        if case["source"] not in corpus:
            problems.append(f"case {index}: thiếu file nguồn {case['source']}")
            continue
        if anchor not in corpus[case["source"]]:
            problems.append(
                f"case {index}: anchor có trong corpus nhưng KHÔNG ở {case['source']}"
            )
            continue

        dataset.append(
            {
                "id": f"gd-{index:02d}",
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "expected_context": case["anchor"],
                "source_file": case["source"],
            }
        )

    if problems:
        print("KHÔNG ghi file, các case sau chưa kiểm chứng được:", file=sys.stderr)
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)
        raise SystemExit(1)

    return dataset


if __name__ == "__main__":
    dataset = build()
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    by_source: dict[str, int] = {}
    for case in dataset:
        by_source[case["source_file"]] = by_source.get(case["source_file"], 0) + 1

    print(f"Đã kiểm chứng {len(dataset)}/{len(CASES)} case -> {OUTPUT}")
    print("\nPhân bố theo tài liệu:")
    for source, count in sorted(by_source.items(), key=lambda item: -item[1]):
        print(f"  {count}x  {source}")
