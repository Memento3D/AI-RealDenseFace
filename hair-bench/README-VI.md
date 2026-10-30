> Bản vá lưu trữ/Ctrl+C: đọc `HOTFIX-1.0.1-VI.md` trước. Chính sách dung lượng mới và chế độ Run offline trong tài liệu đó thay cho hành vi 1.0.

# HairBench 1.0.1 — thử model tạo tóc trên đầu gốc

Bản thử dành cho Legion Ryzen 7 8745H, RAM 16 GB, RTX 4050 Laptop. Phải đo VRAM thực bằng `Inspect`; không suy ra dung lượng trống từ ổ SSD 512 GB. Chạy inference trọng số có sẵn, không train lại.

Mục tiêu của đợt này là xác định model nào chạy được và tạo khối tóc có ích. Không tái tạo hoặc tô lại khuôn mặt. Không tự nhận đầu ra đã đủ điều kiện in.

## Hai ứng viên chạy được bằng bộ điều phối

| ID | Nguồn chính thức | Vai trò và thiết lập lượt đầu |
|---|---|---|
| triposr | https://github.com/VAST-AI-Research/TripoSR | Baseline tạo vật thể 3D, không chuyên tóc. Ảnh vùng tóc trên nền xám; batch 1, chunk 1024, mesh resolution 128. Trích mặt đẳng trị trên CPU để tránh biên dịch torchmcubes CUDA. |
| hunyuan-mini | https://github.com/Tencent-Hunyuan/Hunyuan3D-2 và https://huggingface.co/tencent/Hunyuan3D-2mini | Baseline tạo vật thể 3D, không chuyên tóc. Chỉ shape, ảnh tóc RGBA; fp16, CPU offload, 20 bước, octree 128, không chạy texture/rasterizer. |
| difflocks | https://github.com/Meshcapade/difflocks | Model chuyên tóc, đưa vào danh sách đối chiếu nhưng **chưa có runner inference**. Trọng số yêu cầu đăng ký tại https://difflocks.is.tue.mpg.de/ ; CUDA kernels và VRAM 6 GB chưa kiểm chứng. Không tải hoặc dùng tài khoản thay bạn. |

TripoSR công bố khoảng 6 GB ở thiết lập mặc định. Hunyuan công bố khoảng 6 GB cho shape generation; con số này không phải cam kết cho riêng adapter mini hoặc chiếc laptop này. RAM CPU cũng quan trọng: mini còn có bộ mã hóa ảnh lớn, WSL thường chỉ được cấp một phần RAM Windows. CPU offload giảm VRAM và có thể làm chậm. Không tự sửa `.wslconfig`, cài driver hay bật tính năng Windows.

Chọn hai baseline mesh trước vì output mesh thuận tiện cho bước ghép/in hơn output sợi tóc. Đây là phép thử khả thi, không có bằng chứng chúng sẽ tạo tóc tốt trên mọi ảnh crop; có thể sinh khối đặc, thiếu gáy hoặc sai hướng. Chỉ được quyết định giữ model sau khi xem kết quả thật.

## Giải nén và chạy

Giải nén toàn bộ ZIP vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Sau giải nén có `RUN-HAIR-BENCH.ps1`, `OPEN-HAIR-BENCH.ps1`, `hair-bench\scripts\control.py` ngay dưới project. Không cần thư mục `Memento3D-Figurine-V3`.

Bước 1, kiểm tra phần cứng, không tải model:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-BENCH.ps1 -Mode Inspect
```

Cần một distro WSL2 đã có `python3`, `pip`, `git` và thấy GPU qua `nvidia-smi`. Nếu có nhiều distro, thêm `-Distro Ubuntu` (thay bằng tên thật từ `wsl -l -v`). Nếu WSL hoặc GPU chưa sẵn sàng, dừng ở đây và xem báo lỗi; kit không âm thầm thay cấu hình hệ thống.

Bước 2, cài và chạy tuần tự hai model:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-BENCH.ps1 -Mode All
```

Mặc định dùng workspace cũ đã biết của bạn. Bộ chuẩn bị tìm result V2.1/V2.2 trực tiếp trong `runs\*` có cùng workspace hash và có đủ `shared\source_reference.glb`, `shared\view_0.labels.png`, `checksums.json`, `profile.used.json`. **Không dùng** khuôn mặt đã tô màu đồng nhất trong `canonical_reference.glb` cũ.

ZIP có sẵn `reference04` chứa đầu gốc và mask đã đối chiếu hash từ dữ liệu 04 trước đó của bạn, dùng dự phòng khi không tìm được result trong `runs`. Chỉ dùng khi workspace hash khớp chính xác; không áp dụng đầu này sang người khác. Không bundle workspace ảnh 14 MB.

Nếu result nằm chỗ khác, chỉ rõ thư mục thật:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-BENCH.ps1 -Mode All -Workspace "DUONG_DAN_THAT\head.workspace.html" -MaskRun "THU_MUC_RESULT_THAT"
```

Hai đường dẫn viết hoa ở trên là chỗ bạn thay, không phải thư mục cần tạo. Nếu cần chỉ cài trước: `-Mode Setup`. Nếu đã chuẩn bị subject, thử lại không crop/tạo đầu mới:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-BENCH.ps1 -Mode Run -SubjectDir ".\runs\hair_bench_THOI_GIAN\subject"
```

Có thể chỉ chạy một model bằng `-Models triposr` hoặc `-Models hunyuan-mini`. Các lệnh chỉ Bypass trong tiến trình PowerShell được gọi, không đổi execution policy toàn máy.

## Xem so sánh

```powershell
powershell -ExecutionPolicy Bypass -File .\OPEN-HAIR-BENCH.ps1
```

Chọn từng kết quả trong HTML để xem trên cùng đầu nguồn. Cửa sổ PowerShell phải mở để server local chạy. `SubjectDir` khác có thể truyền bằng tham số như lệnh Run.

Viewer có sẵn trong ZIP, không phụ thuộc Hair Studio cũ. Python CPU `.venv` hiện có cần numpy/scipy/trimesh/Pillow (những thư viện pipeline cũ đã dùng). Model GPU nằm ở môi trường WSL riêng, không cài vào `.venv` này.

**Fit ở viewer là căn khung theo kích thước và vị trí sơ bộ**, có thanh đổi cỡ, độ cao, hướng. Không phải tối ưu tiếp xúc da đầu, không dùng hairline nhiều góc, không chứng nhận in được; tóc có thể xuyên hoặc hở đầu. Chưa thể tuyên bố đã hoàn thành mục tiêu sản phẩm không cần người can thiệp. Lượt thử này dùng để chọn nguồn tóc trước khi triển khai bước fit tự động có kiểm tra va chạm, độ dày và cắt lớp. Việc mặc định tự căn khung không được dùng thay kết quả fit hình học chính xác.

## Output và cách đọc

Trong `runs\hair_bench_<timestamp>`:

- `report.html`, `report.json`, `hardware.json`: phần cứng và trạng thái từng model.
- `<model>.log`: cài đặt và lỗi thực tế; `<model>\result.json`: thời gian và peak VRAM allocator PyTorch nếu inference chạy được. Peak này không bao gồm toàn bộ driver/Windows.
- `<model>\native.glb`, `native.ply`: hình học native, chưa đặt đúng tỉ lệ vật lý trên đầu.
- `subject\portrait.png`, `hair_rgb.png`, `hair_rgba.png`: ảnh điều kiện dùng thật; chỉ ảnh chính diện trong lượt baseline này. Không giả vờ hai model đã dùng cả bốn góc.
- `subject\original_face.glb`: file đầu gốc giữ nguyên byte và SHA-256. File `canonical_reference.glb` trong **viewer mới** chỉ là bản sao byte của file gốc để tương thích tên đường dẫn, không hề canonical hóa màu.
- `subject\viewer\assets\hair`: chỉ các kết quả model thành công. Tên file không thể hiện điểm chất lượng.

`SETUP_IMPORT_CHECKED` chỉ là cài đặt/import thành công, chưa inference. `INFERRED_NATIVE_MESH` là có mesh thật, chưa fit/in. `FAILED`/`FAILED_OR_BLOCKED` có nguyên nhân và log. `DEFERRED_GATED_UNVERIFIED_6GB` là chưa chạy, không tính là thắng. Model lỗi sẽ nhường lượt cho model tiếp theo. Hết thời gian 30 phút inference thì dừng nhóm tiến trình model đó. Setup có timeout 60 phút mỗi lệnh, phụ thuộc tốc độ mạng.

Đánh giá sau lượt chạy: (1) chạy được, thời gian và bộ nhớ, (2) dáng tóc chính diện/nghiêng/gáy so với ảnh gốc, (3) chân tóc/mai/gáy có hợp lý, (4) tiếp xúc đầu và khối tóc, (5) lỗi mesh. `watertight` trong report chưa đủ để đánh giá in resin. Mọi `ready_for_machine` vẫn là false.

## Bộ nhớ, tải và môi trường

Mỗi model có Python 3.10, PyTorch 2.5.0 CUDA 12.1 riêng ở `~/.local/share/memento-hairbench-v1` trong WSL. Git commit và revision trọng số được cố định trong `models.lock.json`. TripoSR tải thêm DINO config đã cố định revision; Hunyuan chỉ tải config + trọng số fp16 của mini, không snapshot toàn bộ repo 25 GB, không tải texture model.

Lần đầu vẫn tải nhiều GB; cache/two Python environments cần thêm dung lượng. Tối thiểu chính sách bộ điều phối là 20 GiB trống, nhưng nên có ít nhất 35–40 GiB cho cả hai model, môi trường và cache. Đây là mức dự phòng của kit, không phải số liệu benchmark. Đóng game/Blender/ứng dụng GPU trước khi thử. Giới hạn 4500 MiB VRAM trống chỉ là điều kiện thử, không bảo đảm tránh OOM. Bộ điều phối dùng GPU số 0.

Không cài MonoHair/Im2Haircut/HairStep mặc định: dependency/toolchain cũ hoặc phần cứng thử nghiệm khác cần kiểm chứng riêng. Không tự tải trọng số DiffLocks có tài khoản. Không bundle trọng số hoặc source repo bên thứ ba trong ZIP; setup clone từ nguồn chính thức. Các model và viewer vendor giữ giấy phép riêng từ nguồn.

## Đã kiểm tra ở môi trường phát triển

Xem `VALIDATION.json`. Đã chạy kiểm tra CPU, tách ảnh và đối chiếu hash bằng workspace thật, thử HTTP/viewer/export bằng mesh fixture tổng hợp và kiểm tra tương thích resolver dependency Python 3.10. **Chưa chạy GPU inference; chưa thực thi PowerShell/WSL trên Windows**. Resolver không tương đương cài/import CUDA thành công. Các file `.resolved.txt` chỉ ghi kết quả resolver PyPI; installer dùng requirements chính sau khi cài torch từ CUDA 12.1 index, không dùng CUDA transitive locks PyPI thay thế.

Nguồn tham khảo phần cứng model:

- https://github.com/VAST-AI-Research/TripoSR
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/docs/source/modelzoo.md
- https://huggingface.co/tencent/Hunyuan3D-2mini
- https://github.com/Meshcapade/difflocks

Test CPU, trong môi trường đã có dependency CPU:

```text
python -m unittest discover -s hair-bench/tests -v
```
