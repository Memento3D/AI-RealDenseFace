# Memento3D — TripoSR Raw Probe v1.1.0

Mục tiêu: thử liệu TripoSR tạo được khối đầu/tóc từ ảnh gốc tốt hơn hướng template hiện tại hay không. Đây là nhánh thử riêng; không sửa hoặc thay HeadLab26, RDF hay DocumentAI. Kết quả là đầu/bust sinh từ MỘT ảnh, chưa tách tóc riêng, chưa khớp với mặt RDF, chưa phải STL/3MF sẵn sàng in.

## Cài đặt một lần

Giải nén ZIP vào gốc project Memento3D hiện có. Yêu cầu `.venv` Python3.11, các script phát hiện mặt của Head24/HeadLab26 và task MediaPipe đang dùng. Đợi tác vụ OCR/RDF dùng GPU kết thúc trước khi chạy inference mới.

```powershell
powershell -ExecutionPolicy Bypass -File .\SETUP-TRIPOSR-PROBE.ps1
```

Bộ cài tạo `.venv-triposr`, lấy upstream vào `vendor/TripoSR-probe-source` hoặc nhận lại source đã xác minh trong thư mục cũ, cài PyTorch2.2.2 + torchvision0.17.2 CUDA12.1 bằng wheel và dependencies riêng. Không kích hoạt hoặc cài thêm package vào `.venv`, `.venv-rdf`, các env OCR. Không cài compiler hoặc native torchmcubes. Thay thế có ghi nhãn bằng scikit-image CPU marching cubes.

Cache/temp lúc chạy nằm ở `D:\AI-cache\triposr-probe`; pip không giữ download cache. Env vars chỉ thay trong process script rồi khôi phục. Nếu có `.git/info/exclude`, script thêm các thư mục env/vendor của probe vào local exclude. Không sửa `.gitignore` chung.

Setup có guard 10GiB trống trên ổ project trước khi tạo env: đây là ngân sách dự phòng, không phải số đo dung lượng cài thật. Bản phát hành chưa kiểm trực tiếp Windows/CUDA hoặc resolver cài package trên máy bạn. Nếu cài thất bại, dừng ở lỗi đầu tiên; không cài tiếp vào env OCR để chữa tạm.

## Thử một bộ trước

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-TRIPOSR-PROBE.ps1 -InputDir ".\input\02"
```

Bản này chỉ lấy `front.jpg/jpeg/png/webp` của bộ đó. Không trộn left/right vào một model single-image. Không phụ thuộc bộ02: có thể thay thành01,03,04,05,06 tùy ý.

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-TRIPOSR-PROBE.ps1 -InputDir ".\input\06"
```

Dùng ảnh cụ thể:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-TRIPOSR-PROBE.ps1 -Image ".\input\06\front.jpg"
```

Dùng ảnh portrait đã tự crop sẵn và bỏ bước dò mặt:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-TRIPOSR-PROBE.ps1 -Image "D:\portraits\head.png" -WholeImage
```

**Dùng ảnh RGB gốc. Không dùng front.fit.png có màu xanh/hồng, ảnh chụp màn hình, hoặc source_reference.png đã chiếu lên đầu 3D.** Các dấu vẽ sẽ trở thành tín hiệu đầu vào sai.

Mặc định resolution128/chunk2048 để thử trước. Khi kết quả có triển vọng mới tăng:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-TRIPOSR-PROBE.ps1 -InputDir ".\input\06" -Resolution 192 -ChunkSize 1024
```

Giảm chunk giúp giảm một phần bộ nhớ lúc query field; không bảo đảm xử lý được OOM xảy ra ở bước neural forward hoặc load model. Worker dừng khi không có CUDA, không lặng lẽ chạy neural trên CPU.

## Đầu ra và cách đánh giá

Mỗi lần tạo `runs/triposr_probe_<timestamp>_<id>/` mới, không ghi đè.

1. `observation/crop_overlay.jpg`: khung crop trên ảnh gốc. Đây là margin hình học từ mặt, chưa chứng minh giữ trọn mọi tóc dài/búi.
2. `observation/head_crop.png`: ảnh đầu trước tách nền. Padding trắng chỉ là canvas, không khôi phục tóc ngoài ảnh.
3. `raw/foreground_rgba.png`: kiểm U2Net có xóa nhầm tóc/giữ nền không.
4. `raw/model_input.png`: ảnh thực sự đưa vào TripoSR.
5. `raw/raw_preview.png`: preview bốn góc Z-up của mô hình sinh; camera chưa đăng ký với các ảnh người dùng. Preview painter có thể giảm số faces hiển thị; mesh xuất giữ đầy đủ.
6. `raw/raw_native.glb`, `raw/raw_native.obj`: mesh thô sau extraction, không thêm smooth/repair, đơn vị native chưa xác minh mm.
7. `raw/report.json`: hash nguồn/model, backend, GPU, peak GPU allocation và thời gian từng bước; trạng thái `RAW_PROPOSAL_EXPORTED` không phải pass chất lượng.
8. `raw/review.html`: mở để xem input/preview và tải mesh. Nếu lỗi, xem `report.json` và `traceback.txt`; lỗi chuẩn bị crop nằm ở `observation/failure.json`.

Lần đầu có tải checkpoint và U2Net; thời gian đó được tách riêng với neural/extraction. Không dùng thời gian lần đầu làm tốc độ inference ổn định.

Cần xem riêng: có giữ được tóc dựng cao/hướng mái/bề ngang không; mặt có bị model sinh thành người khác không; phía sau có hợp lý không; tóc và mặt có thể tách để ghép hay không. Worker chưa chấm độ giống hay tự chọn người giống nhất. Nền/rembg, crop, hình dáng AI và extraction có báo cáo riêng để tìm đúng lỗi.

Trước mắt chạy02 rồi06 tuần tự. Nếu cả hai không cho khối tóc tốt hơn baseline, chưa cần tốn thời gian chạy cả batch, tăng resolution hoặc mesh repair. Chưa gửi lệnh in hoặc xuất machine file.

## Nguồn và khác biệt triển khai

- Upstream: https://github.com/VAST-AI-Research/TripoSR
- Code commit: `107cefdc244c39106fa830359024f6a2f1c78871`.
- Model `stabilityai/TripoSR`: revision `5b521936b01fbe1890f6f9baed0254ab6351c04a`.
- DINO `facebook/dino-vitb16` config: revision `f205d5d8e640a89a2b8ef0369670dfc37cc07fc2`. TripoSR checkpoint chứa weights neural; DINO config được lấy đúng revision này.
- PyTorch wheel pairing: https://pytorch.org/get-started/previous-versions/
- CPU isosurface: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes

Wrapper đăng ký một module tương thích `torchmcubes.marching_cubes` trong process riêng, triển khai thực bằng scikit-image. Không sửa file upstream. Đảo trục return z/y/x đúng với bước đảo lại của upstream; test ellipsoid bất đối xứng kiểm vị trí, kích thước và volume dương. Đây là backend khác native, **chưa có thử nghiệm số học đối chiếu native torchmcubes trên cùng neural density**.

U2Net tải qua rembg; hash file local được ghi, chưa có pin độc lập. Manifest upstream tạo khi tải code commit, các lần sau kiểm hash lại trước khi load. Không chứa model weights, ảnh cá nhân hoặc env trong ZIP.

## Bằng chứng phát hành

Xem `verification/triposr-probe-release.json`. CPU tests không thay cho chạy inference thật. Chưa thực thi neural TripoSR, chưa xác minh đủ VRAM RTX4050, chưa kiểm Windows PowerShell runtime và chưa in vật lý ở máy phát hành.

## Sửa lỗi WinError5 lúc setup (v1.0.1)

Bootstrap không còn đổi tên thư mục source. Nếu lần trước đã tải xong vào `vendor/TripoSR-probe.download`, bản này kiểm commit và hash toàn bộ file trong manifest rồi dùng trực tiếp tại chỗ. Worker cũng dùng cùng resolver. Không tải lại source hoặc model chỉ vì lỗi đổi tên.

Giải nén bản cập nhật vào gốc project rồi chạy lại `SETUP-TRIPOSR-PROBE.ps1`. Không xóa thư mục `.download`, không cần mặc định chạy Administrator hoặc tắt antivirus. Log WinError5 chỉ xác định thao tác đổi tên thất bại, chưa chứng minh nguyên nhân là antivirus, ACL hay thư mục đích đã tồn tại.

Thư mục đích trống được bỏ qua; cây source không có manifest hoặc hash sai được giữ nguyên và báo lỗi rõ. Không tự ghi đè một cây source chưa xác minh. Các lần cài mới dùng đường dẫn ổn định `vendor/TripoSR-probe-source` và không có bước rename thư mục.

## V1.1 — kiểm tra mặt trước khi ghép tóc

Giải nén đè code cập nhật vào thư mục gốc. Giữ nguyên env, cache và run cũ; không cần setup lại cho thay đổi này.

Chỉ kiểm tra kết quả đã có (CPU, không tải model): thay đường dẫn ví dụ bằng run thật của bạn.

```powershell
powershell -ExecutionPolicy Bypass -File .\REVIEW-TRIPOSR-PROBE.ps1 -ProbeRun ".\runs\triposr_probe_TEN_RUN_THAT"
```

Đối chứng độ phân giải, chạy tuần tự trong cùng một lượt, dùng chung một latent field:

```powershell
powershell -ExecutionPolicy Bypass -File .\REVIEW-TRIPOSR-PROBE.ps1 -ProbeRun ".\runs\triposr_probe_TEN_RUN_THAT" -CompareResolutions "128,192,256"
```

Lệnh kiểm tra hash GLB và model_input với report trước khi làm; xuất thư mục `runs/triposr_review_*` mới. Không sửa kết quả gốc. Mở `audit/review.html` để so bản màu không đèn với bản xám có đèn. Rasterizer có z-buffer và vẽ toàn bộ tam giác; camera cùng nhau trong hai chế độ, chưa đăng ký với camera ảnh gốc. Shading xám dùng normal từng mặt nên thấy faceting; không coi đó là phép đo likeness.

Chế độ so sánh dùng lại chính pixel `raw/model_input.png`, bỏ crop/rembg, load model từ cache theo pin cũ, chạy neural forward đúng một lần rồi trích xuất 128/192/256 với cùng ngưỡng density 25. Đây là thay độ phân giải trích xuất, không phải model mới và không bổ sung dữ liệu khuôn mặt đã thiếu. Không cam kết tăng độ phân giải sẽ giống hơn. Xem từng bản, không tự động chọn bản nhiều polygon nhất. Có thể thiếu VRAM ở mức cao; khi CUDA OOM, các bản đã xuất vẫn giữ lại và report ghi PARTIAL, exit code khác 0. Các lỗi khác dừng và ghi traceback.

Nếu lần đầu đã có weight hoàn chỉnh, Hugging Face dùng cache; không xóa cache hoặc tạo env mới. Không chạy song song với tác vụ GPU DocumentAI.

**Chưa thực hiện ghép tóc/RDF trong V1.1.** Mesh TripoSR là cả đầu liền khối; màu tóc không đủ để xác định chân tóc 3D đáng tin cậy. Cần GLB mặt RDF cùng người để kiểm tra căn hệ tọa độ và vùng tiếp giáp. Giữ mặt RDF riêng, không kéo méo mặt theo đầu TripoSR và không dùng mặt nạ màu đen để tự cắt một bộ tóc giả rồi báo thành công. Bản thử này không làm mượt/repair toàn đầu, không khẳng định cá nhân hóa hay sẵn sàng in.
