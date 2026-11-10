# Đánh giá AI Production trên máy local — 03/11/2026

## Kết luận

Ưu tiên thử nhánh ghép mặt vào tượng mẫu của team cho demo tốt nghiệp, giữ nhánh đầu trọc làm đối chứng/dự phòng. Chưa có bằng chứng pipeline mới tạo mặt giống hơn RDF hiện tại hoặc chạy end-to-end trên laptop. Trở ngại hiện tại gồm checkout Windows, đường dẫn môi trường, asset bàn giao và dependency; không phải đã xác định GPU thiếu VRAM.

## Code đã lấy về và vị trí thực tế

- Repo Windows: `D:/FPTU/07/EXE201/ai-3d-generation/Memnoto3D-AI-Production`.
- Fetch origin thành công, remote main mới: `23b32f5b7e039b910b6741e0a13690436faaf95c`, thông điệp `Replace legacy source with src_ai_mem workspace`.
- Checkout/fast-forward trên Windows thất bại vì component `new_face ` có dấu cách cuối tên. Sparse checkout không khắc phục được lỗi kiểm tra đường dẫn. Không tắt `core.protectNTFS`.
- Đã tạo checkout Linux nguyên bản tại `/home/indows/memento3d-production-review-23b32f5` trong Ubuntu-22.04 bằng shallow fetch từ objects đã có trên máy; HEAD đúng commit trên, worktree sạch sau checkout. Remote origin được đặt về repository của team. Chưa cấu hình SSH/auth hoặc tracking để hứa rằng pull trực tiếp trong WSL sẽ hoạt động.
- Repo Windows đã trả về nhánh ban đầu `feature/DonVT-local-gpu-pipeline`, HEAD `0623718`; sparse checkout đã tắt. Các untracked `downloaded-model.glb` và `runs/api/` giữ nguyên. Main Windows chưa được cập nhật.
- Bản source chỉ phục vụ đọc/test trên Windows: `runs/experiments/ai-production-review-01/source-windows-review`. Tên thư mục được chuẩn hóa thành `new_face`, **bytes của 675 file giữ nguyên**; 440 đường dẫn đổi tên được ghi trong `source-review-manifest.json`. Các tham chiếu nội bộ có dấu cách/Linux không bị sửa, nên đây chưa phải bản deploy Windows.
- Archive source-only `source-23b32f5.tar.gz` lấy từ git archive HEAD; không chứa ảnh/weights/env hoặc output local. Không đưa toàn bộ thư mục runs hay archive này vào Git.

## Đọc luồng xử lý

Workspace có nhiều pipeline, không phải một service production thống nhất:

1. `scripts/run_face_on_glb.py` là đường gần mục tiêu tốt nghiệp nhất: GLB mẫu + folder có front/left/right → chuẩn bị vùng đầu bằng HeadBodySplit → quality gate + MICA → chiếu màu → ghép vào GLB → review. Mặc định `source-mode=front`; không mặc định mọi bước đều dùng ba ảnh để phục hồi hình học.
2. Có template ready và khớp checksum GLB: chuẩn bị binding cho subject mới rồi thay patch mặt, giữ phần ngoài patch/material theo kiểm tra. Template không phải dữ liệu generic áp cho mọi GLB hoặc mọi đầu nguồn.
3. Không có template: Blender snapshot/render đầu → nhận diện mặt trên render → mapping/warp mặt → atlas ảnh → validation. Nhánh tự động hiện đòi GLB một mesh, đứng thẳng, mặt hướng -Y, có mặt nhìn rõ. Có thể trả `template_required` hoặc `needs_visual_review`; không được coi là đã hoàn toàn tự động cho mọi GLB.
4. `new_face /AI-RealDenseFace/tools/headapi` là API đầu RDF riêng. API này chưa tự gọi luồng ghép mặt vào tượng ở root. Đổi sang API của repo mới không tự động có body.
5. Body là nhánh SMPL-X fitting từ ảnh toàn thân; không cần chạy nhánh này nếu đã có tượng tốt nghiệp mẫu. `chibi/assemble.py` là nhánh ghép scene đầu-thân khác, không đồng nghĩa boolean thành một khối in kín.

Điểm mạnh phù hợp demo: tận dụng tóc/mũ/quần áo/body mẫu; tập trung cá nhân hóa mặt, giữ các vùng được bảo vệ và có checksum/roundtrip validation. Điểm giới hạn: tóc giữ từ template, ảnh mặt nhỏ/ngược sáng/kính vẫn hạn chế thông tin; ghép đầu không tự cải thiện độ giống và mesh integrity không chứng minh in được.

## Phần cứng và môi trường thực đo

- NVIDIA RTX 4050 Laptop: 6141 MiB VRAM tổng, 5921 MiB trống ở thời điểm truy vấn; driver 560.92. Không thực hiện CUDA inference benchmark ở lượt này.
- D: còn khoảng 24,0 GB ở thời điểm preflight. Tránh cài thêm cả bộ Hunyuan/SMPL-X/tóc cùng lúc; số trống là snapshot, không phải peak disk requirement.
- Blender Windows xác nhận chạy `--version`: **5.2.3 LTS**, ở `C:/Program Files/Blender Foundation/Blender 5.2/blender.exe`.
- WSL Ubuntu-22.04 ban đầu đang tắt, có Python hệ thống 3.10.12; Face/RDF khai báo Python >=3.11,<3.13. Miniforge base Python 3.14.10 không đáp ứng khoảng đó, thiếu NumPy/Pillow. Không sửa các env im2haircut/ml-depth-pro đã có.
- Sau khi khởi động WSL, RAM Windows có lúc còn khoảng 0,93 GB. Đã tắt riêng distro vừa mở, sau đó đo được khoảng 4,68 GB trống tại preflight. Không chạy inference với RAM thấp, không đóng ứng dụng của người dùng.
- Các test chạy tuần tự trong process có guard: bắt đầu >=2 GiB trống, ngừng dưới 768 MiB hoặc timeout 180 giây. Không đo peak RAM/VRAM.

## Asset và dependency

Có thể tái sử dụng có kiểm pin, không cần tải lại ngay:

- `D:/AI-cache/mediapipe/face_landmarker.task` và SCRFD detector trong `D:/AI-cache/mica`.
- MICA checkpoint khoảng 503 MB và FLAME2020 đã có trong vendor của repo Head Reconstruction; preflight ghi hash, không copy asset vào source archive.
- NumPy/Pillow/MediaPipe/OpenCV ở `.venv` Head Reconstruction đúng phiên bản được khai báo cho Face; Torch ở `.venv-rdf` và `.venv-triposr` riêng.

Thiếu hoặc chưa sẵn sàng:

- `GLB/men.glb`, template JSON/annotation/snapshot đã duyệt và chibi body OBJ không có trong checkout. Chưa nhận đường dẫn asset tương đương từ người dùng/member.
- Pose task không thấy ở vị trí cache được kiểm tra. Không kết luận không có ở mọi nơi trên máy.
- MICA config riêng trong repo mới không đi kèm; config Head Reconstruction cũ trỏ `.venv-mica/Scripts/python.exe` hiện không tồn tại. Không âm thầm đổi sang env RDF vì bộ dependency MICA khác.
- CLI root hardcode `new_face /AI-RealDenseFace`, `.venv/bin/python` cho hai env. `--blender` cho đổi executable nhưng chưa đủ để chạy nguyên luồng native Windows.
- Body test còn thiếu smplx/mediapipe trong env Torch được chọn. Đây không phải lý do phải setup Body cho demo template.
- `downloaded-model.glb` đang có trong repo Windows là GLB2, một mesh, 34.120 triangles, khoảng 1,25 MB; mới kiểm JSON container, chưa xác minh có phải tượng phù hợp, hướng mặt hoặc correspondence. Không thay thế men.glb bằng file này một cách mặc định.

## Kiểm tra đã chạy

Tests dùng source review copy có bytes nguyên bản; không phải inference, không phải replay kết quả Linux của member. Không cài package hoặc tải weights ở lượt này.

| Nhóm | Kết quả cuối của lượt kiểm tra | Giải thích |
|---|---|---|
| Root face transfer | 13 tests, 1 error | 12 pass; thiếu GLB/men.glb cho test template |
| Face | 158 tests, PASS | Lượt cấu hình đúng dùng env có MediaPipe |
| HeadBodySplit | 2 tests, 1 error | Test mở np.load mà chưa đóng, Windows không xóa được file tạm; lỗi WinError 32 |
| Body | 10 tests, 2 errors | Env Torch thiếu smplx và mediapipe; 8 tests pass |
| RDF/new_face | 489 tests, 1 failure, 7 errors, 1 skipped | 4 errors từ thư viện tóc STL thiếu; 3 errors im2; 1 failure mocked resume |
| chibi | 4 tests, 1 error | Thiếu body template OBJ |
| Face + RDF architecture | PASS cả hai | Check kiến trúc của đúng source snapshot |
| Head Reconstruction architecture | PASS | Repo hiện tại không đổi code xử lý model |

Lượt đầu dùng env chưa đủ dependency phát sinh lỗi import; đã giữ log và chạy lại các nhóm bị ảnh hưởng với env/path phù hợp. Không cộng các lượt vào một con số pass chung. Chi tiết ở `checks.json`, `configured-checks.json`, `*-unittest.log`, `*-configured-unittest.log`, `*-architecture.log` trong `runs/experiments/ai-production-review-01/`.

## Thứ tự triển khai tiếp được đề xuất

1. Lấy đúng GLB, template/annotation và ảnh mẫu của member để tái lập baseline. Xin cả commit/config và screenshot/report tham chiếu; tránh so kết quả khác input hoặc khác mẫu tượng.
2. Sửa portability trong một branch riêng: rename `new_face ` thành `new_face` và cập nhật references/manifests có chủ đích; cho truyền interpreter theo stage thay hardcode `.venv/bin/python`; cấu hình Blender Windows. Không push sửa lên team nếu chưa được yêu cầu.
3. Chạy trước phần ghép với đầu nguồn đã có và đúng source units/axes, dựng binding cho subject đó. Việc này tách lỗi ghép khỏi lỗi setup MICA, không phải chứng minh end-to-end ảnh → tượng.
4. Sau khi ghép/review đạt, setup riêng MICA worker từ pins/checkpoint sẵn có; chạy một bộ ảnh mới và đo thời gian/RAM/VRAM. Không cài MICA vào env demo RDF hoặc env Hunyuan cũ.
5. Chạy GLB import/roundtrip, bảo toàn vùng ngoài patch, texture, đường nối và review ba góc. Chỉ sau đó mới làm API nối luồng tượng và test web với job mới.
6. In là cổng kiểm tra riêng: units/scale, cổ/patch kín, nonmanifold/intersection, features mỏng, slicer/supports và test resin thực tế.

Không cần mở Digital Salon hoặc thuê cloud ở bước này. Chưa có đo đạc cho thấy GPU 6 GB là blocker; WSL/môi trường/asset là các blocker đã thấy. Nếu native Windows portability được giải quyết, có thể kiểm thử mà không giữ WSL chạy song song.
