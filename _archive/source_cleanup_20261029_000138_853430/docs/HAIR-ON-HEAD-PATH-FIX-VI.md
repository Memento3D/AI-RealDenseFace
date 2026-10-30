# Hair On Head 1.0.1 — đúng bố cục project hiện tại

Thư mục gốc của bạn:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Prompt bạn vừa gửi đang đứng **ngay tại thư mục này**, nên lệnh chạy là `.\RUN-HAIR-ON-HEAD.ps1`. Đường `..\..\RUN-HAIR-ON-HEAD.ps1` chỉ dùng khi đứng bên trong `runs\hair_bench_...`.

## Áp dụng patch

Giải nén **nội dung ZIP thẳng vào thư mục gốc trên**, cho phép thay các file cùng tên của tính năng. Sau giải nén, các đường dẫn chính là:

| Thành phần | Đường dẫn tính từ gốc project |
|---|---|
| Launcher | `RUN-HAIR-ON-HEAD.ps1` |
| Mở lại | `OPEN-HAIR-ON-HEAD.ps1` |
| Lệnh Python | `scripts/run_on_head.py` |
| Lệnh server xem | `scripts/serve_hair_on_head.py` |
| Mã xử lý | `src/memento3d_hair_on_head/` |
| Cấu hình | `configs/hair-on-head.json` |
| Dependencies CPU | `requirements/hair-on-head-cpu.txt` |
| Viewer | `viewer/hair-on-head/` |
| Kiểm tra mã riêng | `hair-on-head-code.json` |

ZIP không có lớp thư mục bọc ngoài và không yêu cầu `hair-on-head/scripts` hoặc `hair-on-head/configs` như gói trước. Không cần di chuyển/xóa các file cũ trước khi áp dụng. Các file `README-VI.md`, `METHODS.md`, `code-checksums.json` cũ không còn được launcher này dùng để tìm mã.

Các helper được đặt riêng bên trong package mới; patch không ghi đè `src/memento3d_head`, các helper dùng chung trong `scripts`, viewer cũ hoặc README tổng của dự án. Hai script entrypoint của riêng tính năng được thay bằng lệnh chuyển tiếp tới mã mới trong src.

## Chạy từ prompt hiện tại của bạn

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-ON-HEAD.ps1
```

Launcher tự tìm benchmark có subject gần nhất trong `runs`. Để chỉ định chính xác lần bạn đã chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-ON-HEAD.ps1 -BenchRun ".\runs\hair_bench_20261028_192116_818"
```

`-BenchRun .` khi đứng ở gốc project cũng được hiểu là tự tìm benchmark gần nhất; khi đứng trong benchmark thì dùng đúng benchmark đó.

Kết quả được tạo tại:

`runs\hair_bench_20261028_192116_818\fitted_hair_<thời gian>`

File xem chính là `head_with_hair.glb`; launcher mở viewer mới đã giữ đúng tọa độ tương đối đầu/tóc. Không chạy lại TripoSR và không kéo/chỉnh tóc bằng tay.

## Xem lại, không dựng lại

```powershell
powershell -ExecutionPolicy Bypass -File .\OPEN-HAIR-ON-HEAD.ps1
```

Hoặc truyền `-FitRun` tới thư mục `fitted_hair_...` cụ thể.

## Đầu vào vẫn dùng dữ liệu đang có

- Python: `.venv\Scripts\python.exe` tại gốc project.
- Workspace: tự chọn file đúng SHA của subject trong `hair-bench/input04/head.workspace.html` hoặc đường dẫn workspace cũ trong `runs/v1_20261025_132459_bbbc53c9/...`.
- Parser mặc định: `D:\AI-cache\face-parsing\resnet18.onnx`.
- Ảnh gáy: tự nhận `input\04\back.jpg` nếu có; ảnh phải qua kiểm tra vùng đầu/hai tai trước khi dùng. `-NoBackImage` chạy với ba ảnh và ghi rõ phần gáy chưa quan sát.
- Có thể chỉ định `-Workspace`, `-ParserModel`, `-BackImage`, `-NoOpen` như trước.

Thuật toán và các giới hạn hình học/in ấn không thay đổi trong patch sửa đường dẫn này. `ready_for_machine=false` vẫn được giữ đến khi có kiểm chứng resin/slicer/in vật lý.

## Kiểm chứng

Đã kiểm tra bố cục phẳng, chạy CLI từ cwd khác và chạy lại pipeline trên bộ dữ liệu thật trong cây thư mục mô phỏng có khoảng trắng. Kiểm tra giải nén trên cây project chứa các file đánh dấu xác nhận không ghi đè module/source chung. Log và kết quả kiểm tra nằm tại `validation/hair-on-head-path-fix/`.

Đây là kiểm tra Python và cấu trúc đường dẫn trên Linux mô phỏng bố cục project; chưa thực thi PowerShell trên Windows trong môi trường này. Patch không tuyên bố đã truy cập hoặc sửa trực tiếp ổ D: của máy bạn.
