# Memento3D — Head Reconstruction & Assembly

Prototype tạo đầu 3D cá nhân từ ảnh và ghép với mẫu tượng tốt nghiệp để xem trên web. Luồng demo hiện tại dùng policy `graduation_modular_preview_v1`: RDF dựng đầu, xử lý texture/da và tóc nền, căn đầu vào template body–tay–mũ, fit tóc preset rồi xuất GLB cho backend.

Tóc hiện là mẫu thư viện được fit theo đầu, chưa phải tái tạo kiểu tóc cá nhân từ ảnh. Output là bản preview; độ giống, các vùng da suy đoán, va chạm, độ dày và khả năng in thực tế vẫn cần kiểm chứng. `ready_for_machine=false` không được đổi thành thành công chỉ vì xuất được GLB.

## 1. Cấu trúc dự án

```text
src/                        Packages: domain, application, pipelines, adapters
tools/headapi/              API, workers, assembly và các công cụ thử nghiệm
tools/customization/        Thử nghiệm tùy chỉnh component
scripts/api/                Launcher API, demo, assembly và test request
scripts/setup/              Cài môi trường theo nhánh
scripts/experiments/        Launcher các phiên bản nghiên cứu
scripts/review/             Công cụ xem kết quả local
scripts/maintenance/        Kiểm tra asset và công cụ dọn dẹp
configs/                    Cấu hình pipeline theo phiên bản
configs/manifests/          Checksum code và nguồn RDF
requirements/               Dependency của các môi trường riêng
tests/                      Kiểm tra hồi quy
docs/guides/                Hướng dẫn chạy và tích hợp
docs/architecture/          Kiến trúc, bản đồ di chuyển file
docs/legacy/                Tài liệu cũ còn được manifest tham chiếu
maintenance/verification/   Báo cáo/checksum evidence cần giữ
assets/ input/ vendor/      Asset, ảnh và model/source local
runs/                       Job/output; một số runtime/asset cũ vẫn được config dùng
```

Các file `requirements*.txt` ở gốc vẫn là đường dẫn cài đặt tương thích. `.env.local`, ảnh người dùng, weights, môi trường và output không được commit. [Danh mục script](scripts/README.md) · [Kiến trúc](docs/architecture/PROJECT-LAYOUT-VI.md) · [Tài liệu](docs/README.md).

## 2. Điều kiện trước khi chạy

- Windows PowerShell; Python **3.11** cho môi trường local đang dùng. Package chính khai báo Python `>=3.11,<3.13`.
- Git; Blender tại đường dẫn `paths.blender` trong config assembly.
- Môi trường RDF/GPU riêng, model/weights và asset đúng phiên bản. Không trộn Torch/CUDA của RDF vào môi trường CPU.
- Các asset assembly đúng SHA-256 trong `configs/headapi-assembly.json`.

**Sau khi clone, chuẩn bị tài nguyên theo [hướng dẫn tải và khôi phục runtime](docs/guides/DOWNLOAD-AND-RESTORE-RUNTIME-VI.md).** Tài nguyên công khai có nguồn tải; asset đã chỉnh của team cần bundle riêng. Hướng dẫn ghi rõ nơi đặt file, checksum, model cần đăng nhập và cách backup nếu mất `D:/AI-cache/`.

| Tài nguyên | Link tải/nhận file | Đặt tại |
|---|---|---|
| Face Landmarker | [Model Google phiên bản 1](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | `D:/AI-cache/mediapipe/face_landmarker.task` |
| Parser ResNet18 ONNX | [Release của tác giả](https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx) | `D:/AI-cache/face-parsing/resnet18.onnx`; dùng `scripts/setup_face_parser.py --output <path>` để kiểm hash |
| RDF source và weights | [Source chính thức](https://github.com/LinzhouLi/RealDenseFace), [gói Drive tại README pinned](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view?usp=sharing), [Hugging Face tác giả](https://huggingface.co/LinzhouLi/RealDenseFace) | `vendor/RealDenseFace/weights/` theo cây thư mục trong hướng dẫn |
| FLAME 2023 / FLAME2020 resources | [FLAME Downloads — đăng nhập](https://flame.is.tue.mpg.de/download.php), [MICA setup](https://github.com/Zielon/MICA/blob/master/install.sh) | Đúng phiên bản/path trong config; không thay model khác bằng cách đổi tên |
| Body, tóc đã chỉnh và reference data | Bundle riêng của team; **chưa có URL chia sẻ trong repo** | Theo từng `assembly.assets.*.path` và SHA-256 trong config |

`vendor/`, weights, AI-cache và asset local không đi kèm Git. Model có tên giống nhau chưa chắc cùng nội dung; kiểm SHA-256 trước khi dùng, không đổi pin để bỏ qua lỗi. Source RDF đang có patch Windows, nên giữ cả snapshot/diff của bản đang chạy khi backup.

**Phòng mất AI-cache:** giữ bản sao model/asset cần thiết trên ổ ngoài hoặc kho chia sẻ riêng, có manifest checksum; xem [quy trình backup và phục hồi](docs/guides/DOWNLOAD-AND-RESTORE-RUNTIME-VI.md#5-backup-để-tránh-mất-ai-cache). Hướng dẫn chưa đồng nghĩa một bản backup cloud đã được tạo hoặc đã thử phục hồi thành công.

Nếu máy đã có môi trường chạy demo, bắt đầu từ bước 4; không tạo lại hoặc nâng cấp môi trường đang hoạt động.

## 3. Setup môi trường trên máy mới

Chạy từ thư mục gốc project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements/headapi.txt
```

Launcher demo/assembly dùng `.venv/Scripts/python.exe`, vì vậy môi trường này cần cả dependency CPU và FastAPI/Uvicorn. Lỗi `No module named uvicorn` cần kiểm tra đúng interpreter đang dùng và cài `requirements/headapi.txt` vào chính môi trường đó.

Môi trường bổ sung được tách riêng:

| Môi trường | Vai trò | Nguồn thiết lập |
|---|---|---|
| `.venv` | API demo/assembly, xử lý CPU và mesh | Các lệnh trên |
| `.venv-rdf` | Worker RDF, Torch/CUDA và renderer | `requirements/rdf-runtime.txt`, `rdf-cuda.txt`, `rdf-build.txt`, `rdf-renderer.txt` và source/weights RDF được team cung cấp |
| `.venv-headapi` | Launcher API head/cap riêng | `scripts/setup/SETUP-HEADAPI.ps1` |
| `.venv-triposr` | Worker/probe riêng khi nhánh được chọn cần tới | `scripts/setup/SETUP-TRIPOSR-PROBE.ps1`; cấu hình `paths.haircap_python` |

Các requirements RDF là baseline theo thành phần, chưa phải bộ cài tự động hoàn chỉnh cho mọi máy Windows. Renderer và solver cục bộ cần build/toolchain phù hợp. Đợt tái cấu trúc này không xác nhận lại cài mới CUDA/renderer; giữ nguyên working environment và pins của team.

Chuẩn bị các đường dẫn trong `configs/headapi-assembly.json`:

1. `paths.cpu_python`, `rdf_python`, `haircap_python`, `rdf_repo`, `blender`.
2. `paths.landmarker`, `parser` và `cache_root` — hiện có đường dẫn local `D:/AI-cache/...`.
3. `assembly.assets`: template, hair library, reference bundle/pose, embedding, semantic masks và UV assets.

Một số asset vẫn nằm trong `runs/` do config hiện tại tham chiếu trực tiếp. Giữ các file đó cho tới khi có migration asset riêng; không xóa toàn bộ `runs/`.

## 4. Cấu hình API key

Nếu chưa có `.env.local`:

```powershell
Copy-Item .env.example .env.local
notepad .env.local
```

Điền:

```dotenv
HEADAPI_API_KEY=<key trùng với cấu hình backend>
HEADAPI_ALLOWED_IMAGE_HOSTS=res.cloudinary.com
```

Không để key rỗng; không đặt key trong frontend hoặc commit `.env.local`. Không ghi đè file này nếu máy đã được cấu hình.

## 5. Kiểm tra và chạy demo assembly

```powershell
.\START-HEADAPI-ASSEMBLY.ps1 -CheckOnly
.\START-HEADAPI-ASSEMBLY.ps1
```

Entry ở gốc chỉ chuyển tiếp tới `scripts/api/START-HEADAPI-ASSEMBLY.ps1`; tham số `-Port` và `-CheckOnly` giữ nguyên.

- Config: `configs/headapi-assembly.json`.
- Policy: `graduation_modular_preview_v1`.
- Cổng mặc định: `8090`; có thể dùng `-Port 8091`.
- Jobs: `runs/headapi-assembly/`.
- Swagger: <http://127.0.0.1:8090/docs>.
- Health: <http://127.0.0.1:8090/health>.

`-CheckOnly` kiểm tra môi trường/key và prerequisites/asset pins; không chạy inference hoặc mở server. Health/preflight đạt chưa chứng minh một lượt image → GLB thành công. Giữ terminal server mở trong lúc frontend/backend gọi API.

Nếu gặp `Assembly requires Blender`, kiểm tra file `paths.blender` có thực sự tồn tại. Lượt kiểm tra khi bổ sung hướng dẫn tải ngày 10/11/2026 không tìm thấy executable ở đường dẫn Blender 5.2 đang cấu hình; xem [cách xử lý](docs/guides/DOWNLOAD-AND-RESTORE-RUNTIME-VI.md#lỗi-hiện-được-phát-hiện-khi-kiểm-tra-docs). Chưa cài hoặc đổi Blender trong đợt cập nhật tài liệu.

## 6. Kết nối backend và kiểm chứng output

Backend gọi AI tại địa chỉ/cổng đã chọn và dùng cùng key trong header `X-API-Key`. Frontend nhận kết quả qua backend theo hợp đồng hiện có.

| Endpoint | Mục đích |
|---|---|
| `GET /health` | Kiểm tra service, readiness và policy |
| `POST /jobs` | Backend tạo job từ URL ảnh theo contract |
| `GET /jobs/{job_id}` | Poll trạng thái job |
| `GET /jobs/{job_id}/outputs/glb` | Nhận GLB của job |
| `POST /v1/head/jobs` | Upload ảnh trực tiếp để kiểm thử local |
| `GET /v1/head/jobs/{job_id}/logs` | Đọc log job local |

Xem request/attempt và hợp đồng chi tiết trong [handoff backend](docs/guides/BE-AI-INTEGRATION-HANDOFF-VI.md), [runbook assembly](docs/ASSEMBLY-API-RUNBOOK-VI.md) và [tham khảo request head/cap](docs/guides/API-REQUEST-REFERENCE-VI.md). Tạo scan/job mới để kiểm chứng pipeline mới; mở output cũ chỉ hiển thị model cũ.

Khi lỗi, xem stage log và report trong thư mục job. Asset pin sai/thiếu phải được xử lý ở nguồn; không trả output giả hoặc tự đổi pin. Cấu hình production, head-only và base-only là các nhánh riêng, không tự động thay thế policy assembly.

## 7. Các launcher khác

```powershell
# Nhánh head + cap, không dùng config assembly
.\scripts\api\START-HEADAPI-DEMO.ps1

# API riêng dùng môi trường .venv-headapi
.\scripts\setup\SETUP-HEADAPI.ps1
.\scripts\api\RUN-HEADAPI.ps1

# Thử nghiệm có phiên bản
.\scripts\experiments\RUN-HEAD24.ps1 -CheckOnly
```

Chỉ chạy setup/probe của nhánh cần dùng. Không bật hai instance trên cùng một cổng. Các profile cũ được giữ để tái lập thử nghiệm, không mặc định coi là pipeline demo hiện tại.

## 8. Kiểm tra mã nguồn

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_architecture.py
```

Lượt kiểm tra docs gần nhất: **602 tests, 1 failure, 4 errors, 1 skipped**; architecture đạt, nhưng preflight assembly dừng vì không tìm thấy Blender tại path cấu hình. Lượt gom JSON trước đó còn gặp thêm failure cache HairStudio, được giữ trong [báo cáo dọn dẹp](docs/architecture/JSON-AND-FOLDER-CLEANUP-VI.md). Không coi suite xanh khi còn failure/error; log lượt docs nằm tại `maintenance/local-cleanup/20261110/unittest-runtime-docs.log`.

## 9. Quy ước lưu trữ

- Cấu hình runtime ở `configs/`; checksum code ở `configs/manifests/`.
- Asset nguồn giữ riêng trong `assets/`; job/output mới trong `runs/`.
- Thông số/report lịch sử được giữ có chọn lọc trong `maintenance/`; dữ liệu cục bộ trong `maintenance/local-cleanup/` được ignore Git.
- Không commit ZIP backup, ảnh người dùng, weights, environments hoặc output. Không dùng Git LFS để che việc đưa output dư vào source.
- Di chuyển file được manifest tham chiếu phải cập nhật path và code checksum có ghi provenance; không đổi model/asset pin âm thầm.
