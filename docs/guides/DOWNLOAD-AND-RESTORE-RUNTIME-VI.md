# Tải tài nguyên và khôi phục runtime

Đối chiếu ngày 10/11/2026 với `configs/headapi-assembly.json`, worker RDF và các nguồn tác giả. Không tự đổi config/model pin trong hướng dẫn này. Link công khai giúp tải lại model; asset đã chỉnh của team cần bản sao riêng. Tải mới chưa được thử inference trên một máy Windows sạch trong đợt cập nhật docs.

## 1. Nguồn tải và nơi đặt

| Thành phần | Nguồn | Vị trí đang dùng / thao tác |
|---|---|---|
| Python 3.11 x64 | [Python cho Windows](https://www.python.org/downloads/windows/) | Chọn dòng 3.11, sau đó kiểm tra `py -3.11 --version`; không chọn phiên bản mới nhất chỉ vì đứng đầu trang |
| Blender | [Trang tải chính thức](https://www.blender.org/download/) | Config hiện trỏ `C:/Program Files/Blender Foundation/Blender 5.2/blender.exe`; dùng bản team đã kiểm chứng. Trình duyệt công cụ chưa xác minh được listing binary 5.2, không suy ra một link installer cụ thể |
| MediaPipe Face Landmarker | [Trang model Google](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker#models), [file phiên bản 1](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | `D:/AI-cache/mediapipe/face_landmarker.task`; kiểm checksum ở mục 2 |
| Parser BiSeNet ResNet18 ONNX | [Repo tác giả](https://github.com/yakhyo/face-parsing), [file ONNX](https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx) | `D:/AI-cache/face-parsing/resnet18.onnx`; dùng downloader có kiểm hash của project |
| Source RealDenseFace | [Repo chính thức](https://github.com/LinzhouLi/RealDenseFace), [README tại commit local](https://github.com/LinzhouLi/RealDenseFace/blob/ebf0063835f520f07fdf5a813b1af9cdc0ae46b5/README.md) | `vendor/RealDenseFace`; baseline commit `ebf0063835f520f07fdf5a813b1af9cdc0ae46b5` |
| Weights RDF và auxiliary assets | [Google Drive từ README tại commit](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view?usp=sharing), [Hugging Face tác giả](https://huggingface.co/LinzhouLi/RealDenseFace) | Đặt các file đúng cây thư mục ở mục 3; không thay bằng ViT-S nếu worker dùng ViT-B |
| FLAME 2023 | [FLAME chính thức](https://flame.is.tue.mpg.de/), [Downloads — cần đăng nhập](https://flame.is.tue.mpg.de/download.php) | Nhận đúng `flame2023.pkl`, đặt vào `vendor/RealDenseFace/weights/flame/`; thực hiện đăng ký/chấp nhận điều khoản tại nguồn |
| FLAME2020 masks cho assembly | [MICA](https://github.com/Zielon/MICA), [hướng dẫn cài của tác giả](https://github.com/Zielon/MICA/blob/master/install.sh), [FLAME Downloads](https://flame.is.tue.mpg.de/download.php) | `vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl`. Cây dữ liệu của script MICA có `data/FLAME2020`; không chỉ lấy file FLAME2023 rồi đổi tên |
| PyTorch CUDA / nvdiffrast | [PyTorch các phiên bản](https://pytorch.org/get-started/previous-versions/), [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | Theo `requirements/rdf-cuda.txt` và `requirements/rdf-renderer.txt`; renderer pin commit riêng. Không nâng version tự động |
| Template/body/tay/mũ, tóc đã fit và tham số tham chiếu | Bộ runtime riêng của team, **chưa có URL chia sẻ được cấu hình trong repository** | Khôi phục đúng path + SHA-256 trong `assembly.assets`; mục 4 liệt kê từng file |

Link Drive/Hugging Face là trang nhận file của tác giả, không phải lời bảo đảm mọi file hiện trên trang đều khớp bản local. Nếu thiếu `face_box.pth` hoặc `flame_assets.npz` trong gói nhận được, giữ checkpoint thất bại và lấy đúng file từ bản backup team; không tự sinh file thay thế. Không tải training data Ava256 chỉ để chạy inference.

## 2. Tải lại hai model trong AI-cache

Chạy từ repository root. Ví dụ tải Face Landmarker vào **file mới** để không ghi đè bản đang chạy:

```powershell
New-Item -ItemType Directory -Force -Path D:/AI-cache/mediapipe | Out-Null
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile 'D:/AI-cache/mediapipe/face_landmarker.download.task'
$expectedTask = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'
if ((Get-FileHash -LiteralPath 'D:/AI-cache/mediapipe/face_landmarker.download.task' -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedTask) { throw 'Face Landmarker checksum mismatch' }
```

Nếu chưa có `face_landmarker.task`, dùng `Move-Item` chuyển file đã kiểm tra sang đúng tên. Nếu đã có, so checksum cả hai; không dùng `-Force` để ghi đè một file khác hash. URL dùng `/1/`, tránh `/latest/` thay nội dung âm thầm.

Parser có downloader sẵn, tự tạo thư mục, kiểm SHA-256 và từ chối ghi đè file sai checksum:

```powershell
.\.venv\Scripts\python.exe scripts/setup_face_parser.py --output D:/AI-cache/face-parsing/resnet18.onnx
```

SHA-256 parser: `0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f`. Nếu mất cả `.venv`, làm bước setup CPU trong README trước khi dùng downloader.

Hai file trên là **dependency cần giữ**, dù đang nằm trong thư mục tên AI-cache. Không xóa chúng như cache tạm.

## 3. Khôi phục RDF source và weights

Chỉ clone vào vị trí chưa tồn tại; không xóa/clone đè vendor đang chạy:

```powershell
git clone https://github.com/LinzhouLi/RealDenseFace.git vendor/RealDenseFace
git -C vendor/RealDenseFace checkout ebf0063835f520f07fdf5a813b1af9cdc0ae46b5
```

Repo local có các chỉnh sửa tương thích Windows; clone commit chỉ khôi phục baseline upstream. Lưu cả snapshot source đang hoạt động và bản diff khi backup. `scripts/patch_rdf_windows.py` chỉ làm patch có phạm vi được kiểm tra của chính script, không được xem là thay thế tất cả thay đổi vendor hiện có.

Sau khi nhận gói weights từ nguồn tác giả và FLAME từ tài khoản của bạn, sắp xếp:

```text
vendor/RealDenseFace/weights/
  facebox/face_box.pth
  flame/flame2023.pkl
  flame/flame_assets.npz
  realdenseface/vitb.pth
```

Worker assembly dùng ViT-B; `vits.pth` không cần để thay thế ViT-B. Hash được worker kiểm tra:

| File | SHA-256 |
|---|---|
| `flame2023.pkl` | `8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0` |
| `flame_assets.npz` | `922cb76f3afef0c2e9801a63991de269a98e6b67b1728232493f851589151dd3` |
| `face_box.pth` | `3fde8561f059dd60d876021cbd2f67d955d5e02f24c25b297913a4115e433b4d` |

ViT-B local được quan sát ngày 10/11/2026: 904.499.225 byte, SHA-256 `3b3bbd026c277aae561aed0047f43bbae2833c000779904f6dd73e97e321c604`. Đây là dấu vân tay bản đang có để phục hồi, **không phải checksum nhà phát hành đã xác minh độc lập hoặc pin mới được thêm vào worker**. Source RDF được đối chiếu bằng `configs/manifests/head24-rdf-source.json`; môi trường RDF cài theo README và requirements riêng.

FLAME và các thành phần có điều khoản riêng cần được nhận/backup theo quyền sử dụng. Không đưa chúng hoặc thông tin tài khoản vào Git/public bundle.

## 4. Asset riêng bắt buộc cho assembly

Không thể tải lại chính xác các file đã chỉnh sau đây chỉ bằng tải template hoặc tóc gốc trên Internet:

| Asset | File cần khôi phục |
|---|---|
| Body/tay/mũ đã chuẩn bị | `runs/hand-hair-assembly-01/attempt-03/assembly.glb` |
| Tóc đã fit/chỉnh | `runs/male-hair-refined-01/input07/cage-lift/head-hair.glb` |
| Reference geometry bundle | `runs/experiments/api-demo-uv-01/case1/rdf/projection_bundle.npz` |
| Reference pose | `runs/graduation-assembly-04/report.json` |
| Embedding | `runs/prod-win-02/input04-production-02/color/embedding.npz` |
| Semantic masks | `vendor/MICA/data/FLAME2020/FLAME_masks/FLAME_masks.pkl` |
| UV assets | `vendor/RealDenseFace/weights/flame/flame_assets.npz` |

SHA-256 chính xác nằm trong `configs/headapi-assembly.json → assembly.assets`, không sao chép pin mới để làm một bản khác trở thành hợp lệ. Reference bundle/embedding có thể chứa dữ liệu cá nhân: lưu trong bundle riêng có giới hạn truy cập, không công khai. Không chia sẻ lại model cần giấy phép nếu chưa có quyền.

## 5. Backup để tránh mất AI-cache

Đây là quy trình thủ công; repository hiện **chưa có** script backup/restore tổng quát được kiểm chứng hoặc một bản backup cloud đã được tạo.

1. Dừng job trước khi snapshot; tạo bundle trên ổ ngoài hoặc nơi đồng bộ riêng, không chỉ ở thư mục khác cùng ổ D.
2. Giữ hai model cache ở mục 2, cây weights mục 3, từng file mục 4 và source vendor đang dùng. Lưu chúng với đường dẫn tương đối dưới bundle; các model ở D:/AI-cache đặt dưới `cache/mediapipe/` và `cache/face-parsing/`.
3. Copy các requirements của project, config không chứa secret, `pip freeze` của từng môi trường, Git commit và vendor diff. Freeze chỉ là inventory, không bảo đảm binary/GPU/build trên máy mới.
4. Ghi `manifest.json` với path trong bundle, path đích, số byte, SHA-256 và nguồn tải của mỗi file. Lấy hash bằng `Get-FileHash -Algorithm SHA256`. Không gồm `.env.local`, API key, tài khoản FLAME, ảnh upload hoặc toàn bộ job ngoài danh sách thực sự cần.
5. Copy bundle sang ổ ngoài/Drive riêng. Repo chưa có link bundle team; bổ sung đường dẫn chia sẻ hạn chế trong kênh nội bộ khi team thực sự upload, không đặt link giả trong README.
6. Thử copy bundle sang một thư mục staging, kiểm lại checksum từng file với bản gốc. Chỉ coi backup dùng được sau kiểm tra này; chưa xóa tài nguyên gốc để thử.

Khi phục hồi, copy file thiếu về đúng đích. Với file đích đã có nhưng khác hash, giữ cả hai để điều tra thay vì ghi đè. Các reference nằm trong runs là dependency đang được pin, không phải output được phép xóa hàng loạt.

Không cần tải lại checkpoint TripoSR chỉ vì env tên `.venv-triposr`: assembly dùng interpreter này cho worker theo config. Chỉ tải model của nhánh khác khi code/policy đó thực sự yêu cầu.

## 6. Kiểm tra sau khi khôi phục

```powershell
.\START-HEADAPI-ASSEMBLY.ps1 -CheckOnly
.\.venv\Scripts\python.exe scripts/check_architecture.py
```

`-CheckOnly` kiểm prerequisites/pins theo preflight, không phải phép xác nhận mọi binary runtime hoặc toàn bộ weights đều tương thích. Tiếp theo chạy API, tạo job mới từ bộ ảnh test của bạn, đối chiếu log/hash/output. Tài liệu nguồn công khai đã mở để đối chiếu; chưa tải lại bộ weights lớn hay thử phục hồi hoàn chỉnh trong đợt viết docs này.

### Lỗi hiện được phát hiện khi kiểm tra docs

Lượt kiểm tra ngày 10/11/2026 không tìm thấy file `C:/Program Files/Blender Foundation/Blender 5.2/blender.exe` đang được config chọn; assembly preflight dừng với `Assembly requires Blender`. Đây là trạng thái kiểm tra thực tế ở lượt này, không phải preflight thành công. Kiểm tra:

```powershell
Test-Path -LiteralPath 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'
Get-Command blender -ErrorAction SilentlyContinue
```

Nhận/cài đúng Blender đã được team kiểm chứng từ nguồn chính thức hoặc bộ cài backup. Nếu máy có Blender ở vị trí khác, xác minh version và cập nhật riêng `paths.blender` trong config được launcher chọn, sau đó chạy lại `-CheckOnly`. Không đổi path sang một file giả để qua kiểm tra. Đợt viết docs này chưa cài Blender hoặc thay đổi config.
