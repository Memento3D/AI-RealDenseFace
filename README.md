# Update v0.13.1 — Mask boundary fix

Xem [UPDATE-v0.13.1.md](UPDATE-v0.13.1.md). Giữ nguyên model/môi trường, chạy lại `--auto-mask` để sửa các khe bị loại nhầm quanh mắt–mũi–miệng.

# Update v0.13.0 — Automatic face masks

Xem [UPDATE-v0.13.md](UPDATE-v0.13.md) để setup model một lần và chạy `optimize-head --auto-mask`. Mask thủ công trở thành công cụ sửa ngoại lệ. Các luồng v0.12 và trước vẫn được giữ.

# Update v0.12.0 — Head texture workspace

Xem [UPDATE-v0.12.md](UPDATE-v0.12.md) để cập nhật bản đầy đủ, chạy một lệnh và review toàn bộ mask/màu/đường nối/vùng thiếu. Luồng mới dùng kết quả RealDenseFace hiện có, chạy CPU, giữ nguyên các run trước.

```powershell
.\.venv\Scripts\python.exe .\scripts\optimize_head.py --head-run runs/rdf_open_20261023_152718_f3ce74/three_photo_7cf7011b
```

# Memento3D Head Reconstruction — 0.3.0

Project độc lập cho nhánh dựng đầu cá nhân hóa từ bộ ảnh. Bộ kiểm ảnh dùng
MediaPipe CPU. Bản 0.2.0 thêm adapter dựng geometry bằng MICA trong môi trường
riêng; chưa xác minh inference MICA/GPU trên Windows, chưa texture hoặc fitting
ba ảnh. Không mang .venv cũ sang. Xem docs/MICA_SETUP_WINDOWS.md cho mốc mới.

## 1. Setup Windows PowerShell

Giải nén thư mục này cạnh Memento3D-AI-check. Mở terminal mới tại project mới:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m memento3d_head config-check --config configs/photo_validation.local.json
```

Không cần activate; dùng Python đúng .venv. Dependency được ghim trực tiếp, chưa có
transitive lock/Windows inference benchmark. Đây là môi trường mới, không sửa môi trường cũ.

## 2. Chuẩn bị Face Landmarker model

Tải task model từ liên kết chính thức trong tài liệu:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models

Đặt ở D:\AI-cache\mediapipe\face_landmarker.task. Pipeline không tự tải model.
Bản model /float16/1 đã được kiểm tra khi tạo project; hash dự kiến bên dưới.
So sánh Get-FileHash với hash này. Nếu khác, dừng và kiểm tra nguồn/version, không
tự sửa hash để bỏ qua lỗi:

```powershell
$modelPath = 'D:\AI-cache\mediapipe\face_landmarker.task'
$modelHash = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
(Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash
.\.venv\Scripts\python.exe -m memento3d_head doctor --model $modelPath --model-sha256 $modelHash
```

MODEL_LOADED chỉ xác nhận khởi tạo task; không phải bằng chứng inference/độ chính xác.
Link trực tiếp đến bản đã kiểm tra và bằng chứng chạy nằm trong docs/VERIFICATION.md.
Không tự tính lại hash để chấp nhận một file đã bị thay đổi ngoài ý muốn.

## 3. Kiểm bộ ba ảnh

Tạo input/ và đặt front.jpg, left.jpg, right.jpg. Khuyến nghị ảnh chụp thật từ cùng
buổi, góc 0° và hai bên khoảng 45°, biểu cảm trung tính, không filter/mirror. Giữ cả tóc,
tai và cằm; v1 chỉ đo mặt, chưa có detector xác nhận tóc/tai đầy đủ.

```powershell
.\.venv\Scripts\python.exe -m memento3d_head validate-photos `
  --front input/front.jpg `
  --left input/left.jpg `
  --right input/right.jpg `
  --source-kind real_capture `
  --model $modelPath `
  --model-sha256 $modelHash `
  --config configs/photo_validation.local.json `
  --output runs/photo_check_001
```

Nếu dùng ảnh AI tạo để test chọn --source-kind synthetic. Không dùng chúng làm
bằng chứng độ giống hoặc hiệu quả multi-view. source-kind chỉ là khai báo.

Output: report.json, 3 ảnh normalized PNG, 3 ảnh overlay landmark có yaw/pitch/roll.
Mỗi lần chạy dùng thư mục output mới. Report/ảnh chứa dữ liệu cá nhân, giữ local;
không commit input/, runs/, weights. Không có tự động xóa ảnh hay upload.

Exit codes: 0 NEEDS_HUMAN_REVIEW; 2 NEEDS_RETAKE; 1 FAILED; 130 CANCELLED.
**Exit 0 không có nghĩa sẵn sàng dựng 3D.** Chưa có review approval workflow.

Ngưỡng góc, độ nét (variance Laplacian trên crop 256), độ sáng trung bình và kích
thước mặt là giá trị thử nghiệm. Chúng không phát hiện đầy đủ bóng sáng, beauty,
che khuất, ảnh giả hoặc khác người. Cần bộ ảnh thật để hiệu chỉnh và test false
accept/reject. Xem docs/ARCHITECTURE.md.

## 4. Tests và boundary check

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_architecture.py
```

Tests dùng evidence giả để kiểm contract/failure, không được gọi là inference thật.
Đọc docs/VERIFICATION.md để biết chính xác những gì đã chạy trong môi trường tạo ZIP.

## 5. Research milestones

Đo CPU validation trên Windows bằng ba ảnh thật. Sau đó thử dựng đầu bằng backend
riêng, đo RAM/VRAM. Chỉ thêm fitting ba góc, texture và tóc khi mỗi bước có bằng
chứng. Không cài MICA/DECA vào .venv này cho đến khi kiểm tương thích dependency.

## Update 0.2.0 — head initialization adapter

New commands: mica-status, mica-pin, mica-doctor, reconstruct-head. Full steps and
limits: **docs/MICA_SETUP_WINDOWS.md**. This adds a real external MICA execution
path, not verified GPU inference in this delivery. Start with cloning/checking
assets; do not install MICA dependencies into the validation .venv.


## Update 0.3.0 — research comparison

New commands: **experiment-head** (one frontal image, research-only quality gate)
and **render-head** (CPU smooth preview of existing OBJ, no inference). See
**docs/SINGLE_IMAGE_EXPERIMENT.md** for upgrade and exact Windows commands.
Existing three-photo approval requirements, MICA config and model pins are
unchanged. No need to reinstall Torch or regenerate model hashes. Smooth preview
changes display only and adds two 90-degree profiles; geometry stays untouched.


## v0.4.0: frontal camera registration

See docs/FRONTAL_REGISTRATION.md. New register-head command estimates an approximate camera and exports photo/mesh landmark overlay without modifying geometry. Seven fit points and one held-out chin diagnostic are not a likeness score. 45 local tests pass; real Windows embedding/MICA mesh integration remains to be verified.


## v0.4.1: three-view camera diagnostics

New register-views command: see docs/THREE_VIEW_REGISTRATION.md. Includes dispatch and embedding batch fixes. Registers three cameras against one unchanged mesh; shape optimization remains unimplemented. 51 local tests pass; real three-view Windows output remains to be checked.


## v0.4.2: shared camera experiment

See docs/SHARED_CAMERA_EXPERIMENT.md. refine-cameras tests shared focal length and robust pose fits without modifying mesh. Includes all preceding fixes. 56 local tests pass. Actual webcam improvement remains to be measured.


## v0.4.3: checkerboard camera comparison

New compare-calibration command reads both experimental camera models, including distortion, and compares their per-view pose fits against register-views. Includes derived numeric webcam calibration profile. See docs/CALIBRATION_COMPARISON.md. Mesh unchanged; no automatic model acceptance or shape/likeness improvement claim. 60 local tests pass; real Windows execution remains to be checked.


## v0.5.0: bounded shared identity fitting

New fit-head command uses three views to produce a conservative eight-parameter FLAME shape candidate while freezing original cameras. See docs/BOUNDED_SHAPE_FIT.md. Source MICA mesh/identity remain untouched. Contour and occlusion handling remain incomplete; no automatic likeness acceptance. 67 local tests pass; real FLAME decoder/Windows results still require verification.


## v0.5.1: dynamic contour diagnostics

New audit-contours command: see docs/DYNAMIC_CONTOUR_AUDIT.md. Exports numbered pose-dependent jaw correspondences and mesh ray self-occlusion for original and candidate, flags angles outside FLAME table support. Does not modify geometry or add unverified contour constraints to fitting. 73 local tests pass; real Windows audit still requires verification.


## v0.6.0: manual visible-contour reference

New prepare-contours and validate-contours commands create an offline annotation editor and check exported source-bound curves. Whole-head outer mesh silhouettes are optional references; no new fitting or automatic anatomical correctness claim. See docs/MANUAL_VISIBLE_CONTOURS.md. 80 local tests, architecture and package-data checks pass. Windows browser interaction remains to be checked.

## New-photo demo (v0.11.0)

Put front.jpg, left.jpg, right.jpg (or PNG/JPEG/WebP) in input/01, then run from
this project root:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m memento3d_head demo --input-dir input/01
```

Review the newly generated landmark images when prompted, enter your name in the
terminal, and the run continues to the offline colored-head preview. Change 01
to any other folder name for a new subject. Runs are isolated; no prior manual
contours or adjusted meshes are reused. Geometry is initialized from the front;
three views supply camera/color information. See docs/demo.md for steps, outputs,
limitations and troubleshooting.
