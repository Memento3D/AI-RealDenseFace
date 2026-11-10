# AI Production — bản chạy Windows đã kiểm chứng

Ngày kiểm tra: 03/11/2026. Upstream: `28ea3e16fbd94537f51b52c382436e308fe02541`, cộng các sửa tương thích được ghi riêng trong `AI-PRODUCTION-WINDOWS-COMPAT.patch`.

## Chạy ngay từ repo AI Production

Mở PowerShell tại `D:/FPTU/07/EXE201/ai-3d-generation/Memnoto3D-AI-Production`:

```powershell
# Xem vị trí source/config và kết quả đã chạy
.\RUN-PRODUCTION-WINDOWS.ps1 -Mode Info

# Kiểm tra MICA nạp được trên GPU
.\RUN-PRODUCTION-WINDOWS.ps1 -Mode Doctor

# Chạy từ ba ảnh Phước đã có đến GLB tượng
.\RUN-PRODUCTION-WINDOWS.ps1 -Mode Images

# Chỉ chạy ghép từ đầu/màu MICA đã có, không inference lại
.\RUN-PRODUCTION-WINDOWS.ps1 -Mode Cached
```

Mỗi lệnh tạo một output mới có ID riêng. Có thể truyền `-OutputDirectory` là thư mục chưa tồn tại. Mode Images nhận `-PhotosDirectory` trỏ folder có đúng một ảnh front, left, right; chất lượng và khả năng chạy với input khác vẫn phải kiểm chứng. Không tự bỏ quality gates hoặc đổi model khi ảnh lỗi.

Không tự mở GUI, không đổi cấu hình web/API, không tự duyệt model để in. `needs_visual_review` là xuất được candidate để xem, không phải xác nhận chất lượng cá nhân hay sản xuất.

## Source, môi trường và file nằm ở đâu

Do upstream vẫn có component `new_face ` không hợp lệ trên Windows, repo Windows gốc vẫn giữ nhánh `feature/DonVT-local-gpu-pipeline`. Launcher ở repo đó gọi bản runtime Windows được chuẩn bị riêng, **không phải checkout main đã được sửa**:

`D:/FPTU/07/EXE201/ai-3d-generation/Memento3D-Head-Reconstruction/runs/prod-win-01/`

- `src/`: source AI Production và runtime assets cần cho case men/Phước, chuẩn hóa thư mục thành new_face.
- `.venv-mica/`: worker riêng Python 3.10.20, Torch 2.5.1+cu121, torchvision 0.20.1+cu121; không sửa env RDF/Hunyuan cũ.
- `src/Face/configs/mica.windows.json`: config mới cho máy này. Năm asset model/detector khớp SHA256 của member, không thay checkpoint.
- Validation/HeadBodySplit/color đang dùng `.venv` CPU của repo Head Reconstruction, lựa chọn source bằng PYTHONPATH theo stage. Không cài hai package memento3d-head trùng tên vào cùng môi trường. Đây là sử dụng thư viện CPU sẵn có, chưa phải bộ cài độc lập hoàn toàn có thể chuyển máy bằng copy folder.
- MICA/FLAME và detector tái sử dụng từ vendor/cache trên máy; config lưu các đường dẫn đó. Không di chuyển/xóa env hoặc assets trước khi rebind và kiểm pins.

Launcher tự chọn đúng working directory và khôi phục PYTHONUTF8 sau khi chạy. Nếu chạy CLI trực tiếp, dùng Python `-X utf8` cùng các interpreter/model/config args; không dùng recipe `.venv/bin/python` của Linux.

## Kết quả đã có

Các đường dẫn sau nằm dưới `runs/prod-win-01/`:

| Lượt | Bằng chứng | Phạm vi |
|---|---|---|
| `attempt-01/` | `run.json`, `validation_report.json`, `face_transferred.glb` | Ghép lại từ đầu/màu member đã có, không inference mới |
| `review.html` | Preview cận mặt/toàn thân của lượt cached | Review trực quan |
| `mica-doctor-01/` | FAILED, giữ nguyên log | InsightFace .pyd rỗng sau build đầu |
| `mica-doctor-02/` | MODEL_LOADED, probe PASS | Nạp MICA và CPU detector, CUDA matmul thật |
| `mica-input04-01/` | GEOMETRY_EXPORTED, probe PASS | Inference mới trên input/04/front.jpg, không phải texture/body pipeline |
| `images-phuoc-01/` | FAILED | Tạo đầu/chiếu màu xong nhưng lỗi font Linux khi xuất review màu |
| `images-phuoc-02/` | FAILED | Đã ghép/kiểm GLB nhưng lỗi encoding khi ghi trang review cuối |
| `images-phuoc-03/` | needs_visual_review, audit PASS | Chạy mới từ ba ảnh → quality → MICA → color → bind → transfer → validation → GLB/review |

Trang xem lượt đầy đủ: `images-phuoc-03/review-rendered.html`. Model: `images-phuoc-03/model.glb`. Audit riêng: `audit-images.json`. Toàn bộ links của trang rendered đã kiểm tra tồn tại.

Thời gian một lượt Phước trên máy này, cộng các stage: **79,61 giây** (head_regions 3,50; quality gồm MICA 10,02; color 9,01; bind 1,52; transfer và validation 55,56). Không bao gồm tải/cài dependency và render review riêng. Không suy ra SLA hoặc hiệu năng cho bộ ảnh khác.

Ảnh Phước được quality gate đánh dấu **CONDITIONAL**, đầu ra cần xem trực quan. MICA geometry chọn front; ba ảnh được kiểm/đăng ký, chế độ màu hiện là **front**, không phải fitting geometry đa góc hoặc tái tạo mặt khuất từ ba ảnh. Tóc/mũ/áo và thiết kế kính thuộc tượng mẫu.

## Sửa tương thích đã áp dụng

- Chuẩn hóa tên folder trong runtime, materialize symlink template bằng payload đã kiểm hash; không hạ/bỏ pin checks.
- CLI cho truyền CPU Python của Face và HeadBodySplit, đặt PYTHONPATH theo stage, dùng Blender Windows.
- Copy template files trên Windows thay symlink cần quyền; xác minh base-template pins trước khi bind, kiểm hash bản copy.
- Font review có lựa chọn Windows/Linux/fallback; JSON/HTML dùng UTF-8, tránh lỗi charmap.
- Cho bỏ preview nặng trong lượt transfer nhưng vẫn giữ export/validation; hai preview cuối render riêng bằng Cycles CPU 8 samples. Trang review không tạo link ảnh render chưa có.
- Mỗi stage CLI có guard >=3 GiB RAM trống để bắt đầu, dừng dưới 1 GiB hoặc quá 600 giây. Cleanup nhắm PID cây process do stage sở hữu, không đóng ứng dụng người dùng. Cached runner dùng timeout 300 giây/stage; model probe dùng 300 giây. Không đo peak RAM toàn máy.

Source manifest: `windows-source-manifest.json`; patch code-only ở docs/AI-PRODUCTION-WINDOWS-COMPAT.patch, không chứa ảnh/weights/env/output. Patch nhắm cây source đã chuẩn hóa tên thư mục, không được apply mù lên nhánh Hunyuan cũ. Cần rename thư mục upstream trước rồi review/cập nhật manifest/source refs. Chưa commit, merge hay push thay đổi lên team.

## Dependency đã tải và vấn đề build

Đã tải Python 3.10 portable và bộ Torch CUDA (wheel Torch khoảng 2,3 GiB), các dependency theo requirements MICA của repo. MSVC Build Tools và Blender đã có trên máy. Không tải lại checkpoint MICA/FLAME/detector và không setup thêm SMPL-X, Hunyuan hoặc tóc.

Build đầu của InsightFace ở đường dẫn cache dài lỗi linker, tạo .pyd 0 byte; retry build có thể bị cache coi là thành công dù binary vẫn rỗng. Sau khi rebuild từ source 0.7.3 trong cache ngắn `D:/AI-cache/memento-mica-win-build`, binary 89.600 bytes có header PE hợp lệ và MICA đã import/nạp/inference được. Không đổi sang wheel không rõ nguồn. Khi build lại, dùng cache ngắn này; không tái sử dụng binary InsightFace hỏng ở cache dài cũ.

`pip check`: PASS. Freeze cuối: `mica-environment.final.freeze.txt`; binary SHA256 và driver ở source manifest. Bộ Torch/torchvision được đối chiếu với [hướng dẫn phiên bản chính thức PyTorch](https://pytorch.org/get-started/previous-versions/).

MICA input04 đo 8,015 giây cho worker process, trong đó inference ghi 0,485 giây; mesh 5.023 vertices/9.976 faces. Torch peak allocated 832.153.088 bytes, reserved 901.775.360 bytes: chỉ là số allocator Torch của lượt đó, không phải tổng VRAM/host RAM. Nạp model đơn thuần không phải inference; chỉ lượt có GEOMETRY_EXPORTED mới được ghi đã suy luận.

## Kiểm tra và giới hạn

- Audit cached và lượt từ ảnh kiểm pins, GLB hash, nguồn head được bind, exit codes, các invariant hình học và ba texture PBR gốc giữ pixel. Roundtrip import đã qua.
- Không phát sinh degenerate triangles hay local intersection theo diagnostic của patch; boundary/nonmanifold không tăng. Phạm vi không bao gồm chứng nhận in hoặc toàn bộ phụ kiện kính.
- Năm regression tests mới về cwd/CLI, template tamper guard, UTF-8 và memory cleanup đều PASS. Runtime root suite: **18 tests, 1 error** vì woman.glb chưa được chuẩn bị trong runtime một-case này. Không tạo model giả hoặc tải thêm mẫu chỉ để làm xanh test.
- Hai architecture check của source Face/RDF PASS. Suite repo Head Reconstruction đã chạy: 535 tests, 1 failure, 4 errors, 1 skipped, thuộc các nhóm còn lại đã ghi trước đó; không báo toàn bộ suite PASS.
- Hình preview cho thấy mặt còn xám/nhạt và vùng mắt thiếu chi tiết. Chưa đo độ giống; không coi successful export là chất lượng đã đạt.
- Chưa có API gọi luồng ghép tượng này, chưa nối web và chưa thử in. HeadAPI RDF trong upstream là luồng đầu riêng.

Sau lượt cài và thử, ổ D còn khoảng 14,5 GiB ở thời điểm kiểm tra; tránh tải thêm các model lớn cùng lúc. Giữ baseline và logs; chuẩn bị case ảnh tốt hơn/đánh giá chất lượng trước khi tối ưu mặt hoặc nối API.
