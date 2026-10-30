# Chạy TripoSR với workspace 04 được khôi phục

Dùng sau khi màn hình đã hiện `TRIPOSR_CACHED_SETUP_READY`.
Lỗi tiếp theo là đường dẫn mặc định trỏ tới workspace trong thư mục runs cũ không còn tồn tại; chưa phải lỗi inference GPU.

Gói này chứa lại **đúng workspace 04 do bạn cung cấp trước đây** tại:
`hair-bench/input04/head.workspace.html`.
SHA256: `0822ea326a0890d895b8a6dcd3caef9ca5fd6cf067fe3531b7d5efc937e0e5f8`.
Hash khớp chính xác với `reference04/report.json`. Không dùng một ảnh hoặc đầu khác thay thế.

Giải nén nội dung ZIP trực tiếp vào:
`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Cho phép thay file mã cùng tên; không xóa `runs`, `.venv` hoặc state/cache/env trong WSL.
Không đặt thêm thư mục bao ngoài ZIP.

Chạy tại thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-TRIPOSR-04.ps1
```

Lệnh kiểm tra hash của workspace rồi gọi HairBench với:
- `Mode Run`
- `Models triposr`
- `Distro Ubuntu-22.04`
- `Workspace` trỏ rõ tới file vừa khôi phục
- `MaskRun` trỏ rõ tới reference04 đã đối chiếu hash

Không chạy lại `RESUME-TRIPOSR-CACHED.ps1`, Setup hay All. Môi trường TripoSR đã qua kiểm tra ở máy bạn được dùng lại; không cài lại dependencies và không khởi động tải Hunyuan. `Run` giữ guard dung lượng/RAM/VRAM và chế độ HF/Transformers offline của Hotfix 1.0.1. Đây không phải sandbox chặn mọi kết nối mạng của mọi thư viện.

Bản launcher chính trong gói cũng chuẩn hóa Windows path thành forward slash trước khi gọi `wslpath` bằng `--exec`. Không sửa weights hoặc checkpoint.

Kết quả mới được tạo trong `runs/hair_bench_<thời gian>`. Nếu thành công, `triposr/result.json` có `status=INFERRED_NATIVE_MESH` và có `native.glb`, `native.ply`. Có mesh không đồng nghĩa tóc đã fit đầu hoặc in được. Nếu bị chặn/lỗi thì xem report và triposr.log mới; dữ liệu môi trường cũ vẫn được giữ.

Kiểm chứng phát triển: chạy chuẩn bị input CPU với workspace khôi phục + mask/reference chính xác, kiểm tra SHA mặt không đổi và ảnh crop tóc. Chưa chạy inference GPU hay thực thi PowerShell/WSL trên máy người dùng trong môi trường phát triển.
