# BTVN#3 — Agent đặt vé máy bay bằng LangChain / LangGraph

Bài tập môn Kỹ thuật xây dựng Agentic AI: xây dựng agent đặt vé với ba mẫu thiết kế **ReAct**, **Plan-then-Execute** và **Hybrid**, cùng lớp Harness kiểm soát thực thi.

Hai chế độ chạy:

- `offline`: model mock theo kịch bản, không cần API key.
- `api`: LLM thật qua `google-genai`, `openai` hoặc `openai-compatible`.

Ở cả hai chế độ, tìm chuyến bay, giữ chỗ và thanh toán đều dùng dữ liệu/tool mock trong bộ nhớ. Mỗi lượt chạy tạo database mới.

## Cấu trúc mã nguồn

| File | Vai trò |
| --- | --- |
| `flight_harness.py` | Ràng buộc, database mock, kiểm quyền, kiểm hoàn thành và bàn giao |
| `agents.py` | ReAct, Plan-then-Execute, Hybrid và schema kế hoạch |
| `model_config.py` | Cấu hình provider, phân loại lỗi và che API key trong log |
| `main.py` | CLI demo, trace và duyệt kế hoạch |
| `evaluate_agents.py` | Benchmark 3 mẫu × 4 kịch bản; xuất và so sánh kết quả |
| `test_agents.py` | Kiểm thử Harness, agent và đánh giá offline |
| `test_api.py` | Kiểm thử cấu hình/adapter API bằng HTTP mock |
| `requirements.txt` | Thư viện cho offline |
| `requirements-api.txt` | Thư viện bổ sung cho API |
| `.env.example` | Mẫu cấu hình, không chứa key thật |

Các cụm code có ghi chú `NOTE` giải thích những điểm kiểm soát và xử lý quan trọng.

## Cài đặt trên Windows PowerShell

Yêu cầu Python 3.11 trở lên và Git.

```powershell
git clone https://github.com/Khanh-24520781/BTTH3_AirTicket_Booking_LangChain.git
Set-Location BTTH3_AirTicket_Booking_LangChain
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu dùng phiên bản Python khác từ 3.11 trở lên, thay `py -3.11` bằng lệnh Python tương ứng. Các lệnh bên dưới gọi trực tiếp Python trong `.venv`, không cần activate.

## Chạy offline

```powershell
# ReAct
.\.venv\Scripts\python.exe -X utf8 main.py --mode offline --pattern react --scenario success

# Plan-then-Execute: nhập y/n để duyệt kế hoạch
.\.venv\Scripts\python.exe -X utf8 main.py --mode offline --pattern plan --scenario success

# Hybrid: thử chuyến dự phòng khi chuyến đầu tiên mất ghế
.\.venv\Scripts\python.exe -X utf8 main.py --mode offline --pattern hybrid --scenario contingency

# Menu tương tác
.\.venv\Scripts\python.exe -X utf8 main.py --mode offline
```

Plan-then-Execute và Hybrid mặc định yêu cầu duyệt kế hoạch. `--auto-approve` dành cho demo tool mock khi cần tự duyệt. ReAct thực thi từng tool dưới kiểm soát của Harness.

## Chạy với LLM qua API

Cài thư viện bổ sung:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
```

Nếu chưa có `.env`, copy mẫu rồi chỉnh trên máy:

```powershell
Copy-Item -LiteralPath .env.example -Destination .env
notepad .env
```

Nếu đã có `.env`, chỉnh file hiện có. Điền `LLM_PROVIDER`, `LLM_MODEL` (model API có quyền sử dụng) và `LLM_API_KEY`. Chỉ lưu key trên máy. `LLM_BASE_URL` là endpoint `/v1` khi dùng `openai-compatible`; để trống với hai provider còn lại.

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --mode api --check-config
.\.venv\Scripts\python.exe -X utf8 main.py --mode api --pattern plan --scenario success
```

`--check-config` chỉ kiểm tra cấu hình/khởi tạo SDK, không gọi API và chưa xác thực key/quyền model. Chạy API thực tế chịu quota và chính sách tính phí của provider. Lỗi model được ghi nhận và bàn giao, không tự chuyển sang mock.

## Kịch bản và kiến trúc

| Kịch bản | Tình huống |
| --- | --- |
| `success` | Có chuyến hợp lệ để giữ chỗ và thanh toán |
| `fail` | Không có chuyến đạt cả giờ và giá; cần bàn giao |
| `contingency` | VN122 mất ghế sau tìm kiếm; VN124 là dự phòng |
| `transient_error` | Tìm kiếm lỗi tạm thời ở lần đầu |

Ràng buộc mặc định: SGN → DAD ngày 2026-10-07, khởi hành trước 12:00, giá tối đa 2.000.000 VND, hành khách Nguyen Van A.

Harness lưu ràng buộc dưới dạng dữ liệu, kiểm quyền trước khi tool chạy, nghiệm thu bằng bản ghi trong database mock và tạo bàn giao có trạng thái/lịch sử/lý do. Câu trả lời model không đủ để xác nhận hoàn thành.

- **ReAct:** chọn hành động dựa trên kết quả các tool trước đó.
- **Plan-then-Execute:** lập một kế hoạch, duyệt rồi chạy tuần tự; bàn giao khi không thể tiếp tục.
- **Hybrid:** lập kế hoạch kết hợp thực thi có retry và lập lại kế hoạch khi phát sinh sự cố.

## Đánh giá và kiểm thử

```powershell
# Ma trận offline: 3 mẫu × 4 kịch bản
.\.venv\Scripts\python.exe -X utf8 evaluate_agents.py --mode offline --output-dir outputs/offline

# Benchmark API: cần .env; có gọi LLM thật
.\.venv\Scripts\python.exe -X utf8 evaluate_agents.py --mode api --output-dir outputs/api

# So sánh hai bộ kết quả đã lưu; không gọi LLM
.\.venv\Scripts\python.exe -X utf8 evaluate_agents.py --compare-json outputs/offline/benchmark_results.json outputs/api/benchmark_results.json --output-dir outputs/comparison

# Kiểm thử offline
.\.venv\Scripts\python.exe -X utf8 -m unittest -v test_agents

# Kiểm thử cả adapter API sau khi cài requirements-api.txt; dùng HTTP mock
.\.venv\Scripts\python.exe -X utf8 -m unittest -v test_agents test_api
```

Benchmark tự duyệt kế hoạch, xuất JSON, CSV, log và manifest phiên bản/cấu hình. Bảng đánh giá phân biệt đặt vé thành công, bàn giao đúng kỳ vọng và lỗi model; có số lần gọi model/tool, thời gian và token nếu provider cung cấp. Kết quả offline phản ánh chính sách mock; cần đánh giá API riêng để khảo sát hành vi LLM thật.

## Phạm vi repository

Repo chỉ gồm mã nguồn, kiểm thử, requirements, mẫu cấu hình và hướng dẫn chạy. `.gitignore` dùng danh sách cho phép; `.env`, môi trường ảo, cache, ảnh, báo cáo và kết quả benchmark nằm ngoài repo. Khi thêm file nguồn mới, cập nhật `.gitignore`.
