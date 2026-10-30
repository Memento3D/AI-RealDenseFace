# Memento3D — mốc Pipeline V2.3

- Chạy chính: `powershell -ExecutionPolicy Bypass -File .\RUN-CURRENT.ps1`.
- Launcher gốc/checksum: `RUN-PIPELINE-V23.ps1`, `pipeline-v23-code.json` — giữ nguyên.
- Hướng dẫn/phương pháp: `PIPELINE-V23-START-VI.md`, `PIPELINE-V23-METHODS.md`.
- Cấu hình: `configs/pipeline-v23-04.json`; không thay model pin.
- Workspace hiện tại: `hair-bench/input04/head.workspace.html`.
- Tài nguyên tóc: `assets/hair`; parser: `D:/AI-cache/face-parsing/resnet18.onnx`.
- Kết quả chốt: `runs/pipeline_v23_20261028_232602_556`.
- `src/`: giữ toàn bộ kiến trúc ứng dụng và dựng đầu từ ảnh.
- `scripts/`: giữ module phụ thuộc; tên V2/V21/V22 không đồng nghĩa bỏ được.
- `tests/`: giữ toàn bộ; một số test cần tài nguyên và SDK ngoài gói source.
- `_archive/source_cleanup_.../docs`: tài liệu lịch sử.
- `_archive/source_cleanup_.../launchers`: lệnh thử nghiệm cũ đã tạm ngưng.
- Cache pip có thể tạo lại; không xóa model OCR/Hugging Face trong gói dọn này.

Các launcher đã chuyển vào archive không chạy trực tiếp từ vị trí mới vì dùng
PSScriptRoot làm thư mục project. Hoàn tác để dùng lại; không sửa đường dẫn nguồn
hay tự thay checksum chỉ để launcher cũ chạy từ archive.

Đây là dọn tổ chức, không phải thay đổi thuật toán, refactor toàn bộ src hay
chứng nhận in được. Việc giải phóng GB xử lý cache riêng; di chuyển trên cùng ổ
đĩa không tăng dung lượng trống.
