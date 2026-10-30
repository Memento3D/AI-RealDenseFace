# Setup Windows — main + RealDenseFace (V2.4.1 setup update)

Bản này bổ sung setup, không đổi thuật toán hoặc chất lượng đầu ra V2.4.
Python 3.11 cho cả hai môi trường. Chạy mọi lệnh từ thư mục gốc project,
trừ khi có Push-Location. Không cần activate.

## 1. Máy hiện tại đã chạy được

Không xóa/tạo lại môi trường hoặc chạy toàn bộ phần cài mới. Kiểm tra trước:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_setup.py --environment main
.\.venv-rdf\Scripts\python.exe .\scripts\check_setup.py --environment rdf
```

Lưu snapshot đúng môi trường hiện có để đối chiếu khi chuyển máy:

```powershell
New-Item -ItemType Directory -Path local_setup -Force | Out-Null
.\.venv\Scripts\python.exe -m pip freeze --all | Out-File local_setup/main.freeze.txt -Encoding utf8
.\.venv-rdf\Scripts\python.exe -m pip freeze --all | Out-File local_setup/rdf.freeze.txt -Encoding utf8
.\.venv-rdf\Scripts\python.exe -m pip inspect | Out-File local_setup/rdf.inspect.json -Encoding utf8
.\.venv-rdf\Scripts\python.exe .\scripts\check_setup.py --environment rdf --output local_setup/rdf.check.json
```

Snapshot có thể chứa đường dẫn local; dùng để đối chiếu, không tự coi là lock
portable. Không commit local_setup. Các requirements mới chỉ ghim một số dependency
trực tiếp theo môi trường đã ghi nhận; dependency có khoảng phiên bản và transitive
chưa được khóa hoàn toàn. Chưa kiểm chứng cài sạch Windows/CUDA của bản setup này.

## 2. Cài mới môi trường chính `.venv`

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe .\scripts\check_setup.py --environment main
```

`requirements.txt` cài package editable, vision, observability và parser CPU.
Không cài đồng thời opencv-python vào môi trường này: vision dùng opencv-contrib-python.

## 3. Clone đúng RealDenseFace (chỉ cho máy chưa có vendor)

```powershell
New-Item -ItemType Directory -Path vendor -Force | Out-Null
git -c core.autocrlf=false clone https://github.com/LinzhouLi/RealDenseFace.git vendor/RealDenseFace
git -C vendor/RealDenseFace checkout ebf0063835f520f07fdf5a813b1af9cdc0ae46b5
```

Nếu đã có vendor đang chạy: giữ nguyên, không reset/checkout đè bản sửa Windows.
Bản sửa được cung cấp có kiểm tra commit và nội dung trước khi ghi:

```powershell
.\.venv\Scripts\python.exe .\scripts\patch_rdf_windows.py
```

Sửa duy nhất việc truyền compile_model từ inferencer xuống FaceDetector, để
`--no_compile` không kích hoạt Torch Inductor/Triton ở detector. Idempotent;
không reset các thay đổi vendor khác. Nếu nội dung không đúng mẫu thì dừng.

## 4. Cài mới `.venv-rdf` và build extension

Cần NVIDIA driver tương thích, CUDA Toolkit 12.6 (có nvcc), Visual Studio 2022
Build Tools với Desktop development with C++ và Windows SDK. Mở **Developer
PowerShell for VS 2022** rồi cd về project. Torch có CUDA runtime chưa đủ để
biên dịch extension: vẫn cần toolkit và compiler. Baseline này theo RTX 4050.

```powershell
py -3.11 -m venv .venv-rdf
.\.venv-rdf\Scripts\python.exe -m pip install --upgrade pip
.\.venv-rdf\Scripts\python.exe -m pip install -r requirements/rdf-build.txt
.\.venv-rdf\Scripts\python.exe -m pip install -r requirements/rdf-cuda.txt --index-url https://download.pytorch.org/whl/cu126
.\.venv-rdf\Scripts\python.exe -m pip install -r requirements/rdf-runtime.txt
.\.venv-rdf\Scripts\python.exe -m pip install chumpy==0.70 --no-build-isolation
```

Kiểm tra lệnh nvcc, cl và Torch trước khi build. Nếu CUDA_PATH không đúng vị trí
cài thực tế thì sửa cho phiên terminal này; ví dụ máy Don: D:\CUDA\v12.6.

```powershell
$env:CUDA_PATH = "D:\CUDA\v12.6"
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = "8.9"  # RTX 4050; GPU khác cần kiến trúc phù hợp
$env:MAX_JOBS = "2"
nvcc --version
Get-Command cl
.\.venv-rdf\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\.venv-rdf\Scripts\python.exe -m pip install -r requirements/rdf-renderer.txt --no-build-isolation
.\.venv-rdf\Scripts\python.exe -m pip install -e ./vendor/RealDenseFace/cuda_extensions/flame_solver --no-build-isolation
.\.venv-rdf\Scripts\python.exe -m pip check
.\.venv-rdf\Scripts\python.exe .\scripts\check_setup.py --environment rdf
```

Trong PowerShell, nếu một lệnh cài/build báo lỗi thì dừng ở đó, không tiếp tục
bằng các lệnh sau. Nvdiffrast v0.3.3 được ghim commit để tránh theo HEAD thay đổi;
chưa xác nhận đây chính là revision đã có trên máy Don. Giữ bản đang chạy tốt,
chỉ thử baseline này khi cài môi trường mới. Không cần cài Triton cho luồng eager.
Chumpy phải được import sau `common.fix_chumpy` của upstream trên Python 3.11/NumPy1.26.

## 5. Weights và cache

Tải bộ weights từ link Google Drive trong README của đúng commit upstream:
https://github.com/LinzhouLi/RealDenseFace/tree/ebf0063835f520f07fdf5a813b1af9cdc0ae46b5

Giải nén sao cho có đúng các file dưới đây (không lồng weights/weights):

```text
vendor/RealDenseFace/weights/facebox/face_box.pth
vendor/RealDenseFace/weights/realdenseface/vitb.pth
vendor/RealDenseFace/weights/realdenseface/vits.pth
vendor/RealDenseFace/weights/flame/flame_assets.npz
vendor/RealDenseFace/weights/flame/flame2023.pkl
```

FLAME2023 tải riêng, đăng nhập và đọc điều khoản tại https://flame.is.tue.mpg.de/download.php.
Dùng **FLAME 2023** tương thích pipeline hiện tại; không thay bằng FLAME2020,
không đổi tên bản Open thành bản gốc. Source ZIP không phân phối model này.
Checker đối chiếu hash FLAME/assets với bản đã chạy trước; weights còn lại ghi
hash quan sát được, không tuyên bố đã xác thực nhà phát hành khi chưa có hash chuẩn.

Parser tự tải từ nguồn và SHA256 đã ghim trong source:

```powershell
.\.venv\Scripts\python.exe .\scripts\setup_face_parser.py --output D:/AI-cache/face-parsing/resnet18.onnx
```

Face Landmarker chỉ cần cho CLI kiểm ảnh/registration cũ; xem phần model trong
README.md. Luồng run_v2 hiện không tự áp dụng gate đủ ba góc/độ nét. Có đủ tên
front/left/right chưa chứng minh bộ ảnh đạt chất lượng.

## 6. Chạy và kiểm chứng thực tế

```powershell
.\.venv\Scripts\python.exe .\scripts\check_setup.py --environment main
.\.venv-rdf\Scripts\python.exe .\scripts\check_setup.py --environment rdf
.\.venv\Scripts\python.exe .\scripts\run_v2.py --input-dir input/04 --complete-head
```

Đổi input/04 theo bộ ảnh. Mỗi bộ có front, left, right (jpg/jpeg/png);
back và references/*-glasses là tùy chọn cho review. Chạy lần đầu để xác nhận
CUDA kernels, weights và toàn pipeline hoạt động; checker chỉ kiểm tra setup,
CUDA matmul/import extension, không kiểm chứng inference hay độ giống.
Xem pipeline.report.json và file GLB trong run mới. Tóc/kính V2.4 vẫn là prototype.

## 7. Các lỗi thường gặp

- `configs/model/vitb.yaml` không tồn tại: chạy upstream từ vendor/RealDenseFace,
  hoặc dùng run_v2.py vốn đặt cwd đúng.
- `TritonMissing`: chạy patch_rdf_windows.py và dùng --no_compile khi chạy upstream.
- `nvcc` / `cl` không tìm thấy: kiểm tra toolkit và Developer PowerShell.
- `DLL load failed` / `no kernel image`: extension phải build lại trong đúng Python,
  Torch/CUDA và kiến trúc GPU; không copy site-packages giữa máy.
- `ModuleNotFoundError`: kiểm tra đường dẫn Python; không dùng pip chung chung.
- Nhiều OpenCV distributions cùng môi trường: chúng cùng cung cấp cv2;
  dùng đúng loại của từng requirements, không trộn hai môi trường.

MICA/DECA không phải dependency bắt buộc của V1/V2 RDF. `.venv-mica` là nhánh
nghiên cứu riêng, xem docs/MICA_SETUP_WINDOWS.md.

Nguồn: README/requirements/setup.py tại commit RDF đã ghim; lệnh Torch2.7.1/cu126:
https://pytorch.org/get-started/previous-versions/#v271
