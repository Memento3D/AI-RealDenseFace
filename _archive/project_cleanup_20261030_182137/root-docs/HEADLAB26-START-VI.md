# HeadLab26.1 — chạy độc lập nhiều bộ ảnh

Bản bổ sung cho project đang chạy Head24.5/V2.3. Giải nén **vào thư mục gốc project**, nơi có `src`, `scripts`, `configs`, `.venv` và `vendor`. Không chạy từ thư mục ZIP mới tách rời. Bản này giữ nguyên mã Head24.5 và 63 file runtime V2.3; mã mới nằm trong `tools/headlab26`. Không kèm ảnh, kết quả cá nhân, weights hay môi trường Python.

## 1. Chuẩn bị dữ liệu

Mỗi thư mục độc lập:

```text
input/01/front.jpg
input/01/left.jpg
input/01/right.jpg
input/02/front.png
input/02/left.png
input/02/right.png
...
input/06/front.jpg
input/06/left.jpg
input/06/right.jpg
```

- Hỗ trợ JPG/JPEG/PNG/WebP và EXIF orientation. Không đặt cùng lúc `front.jpg` và `front.png` trong một bộ.
- Front/left/right phải là ba ảnh khác nhau của cùng người. Tên left/right theo quy ước chụp nhất quán; bản này không tự xác minh danh tính hay đổi nhãn hai bên.
- Ảnh toàn thân được tự tìm mặt và crop; ảnh thiếu sáng, mắt bị che, quá nhỏ hoặc nhiều khuôn mặt sẽ có cảnh báo/từ chối riêng. Phóng crop lên 1024 không tạo thêm chi tiết thật.
- `back.jpg` là tùy chọn: ghi nhận/hash để xem lại, **chưa dùng làm camera fit phía sau**. Không tuyên bố đã khôi phục gáy từ ảnh back chưa hiệu chuẩn.
- Thiếu một bộ hoặc lỗi một nhánh không làm hỏng các bộ còn lại.
- Muốn giữ tên ảnh gốc, đặt `headlab.json` trong bộ đó:

```json
{"views":{"front":"IMG_001.jpg","left":"IMG_002.jpg","right":"IMG_003.jpg"}}
```

Không có tham chiếu bắt buộc đến `input/05`, workspace cũ hay một run cố định.

## 2. Lệnh chạy

Mở PowerShell tại thư mục gốc project. Không cần thay ExecutionPolicy toàn hệ thống.

**Kiểm tra đủ file/model trước, không inference:**

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB26.ps1 -Datasets "01-06" -CheckOnly
```

**Khuyến nghị lần đầu: so sánh khuôn mặt của cả sáu bộ, chưa làm tóc/kính/STL:**

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB26.ps1 -Datasets "01-06" -FaceOnly
```

Lệnh này chạy hai nhánh RDF trên cùng crop và cùng dự đoán neural của từng bộ, rồi xuất mặt xám, màu nguồn, màu nền chung và overlay đường mí mắt. Đó là hai thuật toán fit để đối chứng, không phải hai lần huấn luyện model.

**Chạy đầy đủ mặt + các ứng viên tóc + kính + STL cho bộ cụ thể:**

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB26.ps1 -InputDir ".\input\06"
```

**Chọn bộ bất kỳ, hoặc dùng ổ/thư mục input khác:**

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB26.ps1 -Datasets "01,03,06"
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB26.ps1 -InputRoot "D:\MyDatasets" -Datasets "01-06" -FaceOnly
```

Không truyền `-Datasets` thì tìm các thư mục con trực tiếp trong InputRoot. `-AllVariants` xuất tóc cho mọi nhánh fit thành công, tốn thời gian/dung lượng hơn. Mặc định chỉ xuất tóc cho một nhánh; **khi mắt bị kính che/không có semantic support, giữ thứ tự baseline, không dùng một điểm proxy thấp để tuyên bố phục hồi mắt tốt hơn**.

Lệnh Python tương đương nếu không dùng PowerShell:

```powershell
.\.venv\Scripts\python.exe -m tools.headlab26.cli --datasets "01-06" --face-only
```

## 3. Xem kết quả và chạy tiếp

Cuối lệnh in đường dẫn `runs/headlab26_<timestamp>/index.html`. Mở file đó để xem so sánh. Mỗi lần chạy tạo thư mục mới, không ghi đè.

- `summary.csv`: mỗi dòng là một dataset/backend; thời gian và lỗi riêng.
- `batch.report.json`: tổng hợp cấu hình, fingerprint input, trạng thái từng bộ.
- `datasets/XX/<backend>/diagnostics/`: mặt xám, overlay mí mắt xanh lá (MediaPipe) và hồng (FLAME), số pixel mắt/kính và proxy.
- `.../appearance/`: màu nguồn, màu nền chung và vùng da tham chiếu; phép đổi màu không sửa hình học mặt.
- `.../figurine/review.html`: ứng viên tóc/kính, file GLB/STL và các trường hợp bị loại. Không có tóc qua gate vẫn giữ chẩn đoán mặt.
- Từng bước subprocess có `.log` và `.stage.json`, command/exit code/thời gian thật.

Mã thoát 2 là batch có cảnh báo/bộ lỗi; PowerShell vẫn in hướng mở kết quả. Không có nghĩa mọi bộ đều thất bại.

Tái sử dụng crop/fit đã xác minh từ **batch HeadLab26** trước:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB26.ps1 -Datasets "01-06" -ResumeRun ".\runs\headlab26_TEN_RUN_TRUOC"
```

Chỉ tái sử dụng khi ảnh, đường dẫn nguồn, model, policy, code và hash các file cache khớp. Input thay đổi thì chạy lại. Cache bị sửa bị từ chối, không lặng lẽ dùng tiếp. Cache 26.1 tách hash mã crop/semantic/fit; sửa riêng mã tóc không làm mất cache fit. Cache 26.0 dùng key cũ nên cần chạy lại một lần khi nâng lên 26.1. Đây không phải cú pháp resume của Head24 cũ.

## 4. Các nhánh và giới hạn

| Nhánh | Thực hiện | Cách hiểu |
|---|---|---|
| `rdf_baseline` | RDF ViT-B, fit chung identity, pose/FOV riêng cho từng ảnh | Đối chứng cùng budget, không đồng nhất chính xác với mọi preset Head24 cũ |
| `rdf_occlusion` | Giảm ảnh hưởng điểm dự đoán nằm trong kính/tóc/nền, thêm tái trọng số Huber | Giả thuyết thử nghiệm; không suy ra cấu trúc mắt đang bị che |
| `pixel3dmm` | Adapter gọi upstream crop, segmentation, normal/UV prediction, multi-image tracking rồi xuất camera/mesh | Tùy chọn, cần môi trường và weights riêng; chưa xác minh GPU/Windows ở bản phát hành này |
| TRUST | Ước lượng UV albedo từ face + toàn cảnh từng ảnh | Tùy chọn để đối chiếu; không tự coi median UV là màu da chuẩn và không tự áp lên mặt |
| Radial hair | Khớp vỏ tóc có giới hạn, sửa tóc mai cục bộ, kiểm tai/tiếp xúc/crown | Hai asset demo có sẵn trong project, chưa phải thư viện mọi kiểu tóc |
| ARAP hair | ARAP trên đồ thị điều khiển tách lớp, truyền displacement về mesh gốc và hạn chế hướng tam giác | Nhánh hybrid có thể bị loại; kiểm thể tích, tiếp xúc và hướng tam giác vẫn riêng; không chứng nhận self-intersection hoặc khả năng in |

Mặc định dùng cả radial và ARAP để thấy rõ nhánh thành công/bị loại trong cùng lần chạy. Trên replay05 CPU thực tế của bản 26.1, hai radial và hai hybrid ARAP của mẫu 21778 đều xuất được. Mẫu 11256 vẫn bị loại do che tai/vượt ngân sách dịch chuyển; không bỏ gate để ép xuất. Xem `verification/head261-release.json`. Muốn chỉ chạy radial, sao chép profile rồi đặt `figurine.hair_methods` thành `["radial"]`.

Đối với tóc dài/búi, quy tắc che tai 10% của demo tóc ngắn không phải tiêu chuẩn giải phẫu phổ quát. Cần thêm asset và policy riêng trước khi đánh giá các kiểu đó. Bộ dữ liệu đa dạng dùng để phát hiện các giới hạn này, không tự tạo một kiểu tóc bất kỳ từ ảnh.

Mắt kính là hình học tham số ước lượng, mắt bị che có fallback thiết kế được ghi nhãn. Không tuyên bố đã khôi phục mắt từ ảnh có kính. Nếu cần nhận diện mắt đáng tin cậy hơn, nên có ít nhất một ảnh đủ sáng không kính.

## 5. Cấu hình máy và backend tùy chọn

Mặc định dùng môi trường/model sẵn có:

- CPU: `.venv` của project; cần các dependency CPU đã dùng ở Head24.5 (NumPy/SciPy/Pillow/Trimesh/Scikit-image/ONNX Runtime/MediaPipe/Manifold3D...).
- RDF GPU: `.venv-rdf`, `vendor/RealDenseFace` và weights đã pin của project.
- Parser: `D:/AI-cache/face-parsing/resnet18.onnx`.
- Landmarker: `D:/AI-cache/mediapipe/face_landmarker.task`.

Không tải hay cài tự động model mới. Không sửa cache OCR/Hugging Face, không xóa env hay run cũ.

Sao chép `configs/headlab26.json` thành tên riêng, sửa các trường `paths` cần thiết, truyền `-Profile`. Mọi đường dẫn tương đối tính từ gốc project; dùng `/` hoặc `\\` đúng JSON cho đường dẫn Windows. Nếu chỉnh một file thuộc manifest phát hành, integrity check sẽ dừng; **sửa bản sao profile**, không sửa file mặc định đã pin.

### Pixel3DMM

Cài upstream riêng theo README chính thức: <https://github.com/SimonGiebenhain/pixel3dmm>. Chuẩn bị dependencies, normal/UV checkpoints, FLAME2023 và tài nguyên segmentation theo upstream. Không trộn môi trường RDF hiện tại.

Trong bản sao profile, đặt `pixel3dmm.python`, `pixel3dmm.repo`; thêm `required_files` cho tài nguyên cục bộ. Sau đó:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB26.ps1 -Datasets "01,05,06" -Backends "rdf_baseline,rdf_occlusion,pixel3dmm" -Profile ".\configs\headlab26-local.json" -FaceOnly
```

Adapter gọi thật các script upstream bằng argv, kiểm exit code; dùng `is_discontinuous=True`, camera riêng, `use_flame2023=True`, `ignore_mica=True`, batch tối đa 3 ảnh. Camera xuất qua hợp đồng riêng; overlay độc lập vẫn cần kiểm. Kiểm tra file hiện diện chưa chứng minh CUDA/dependencies tương thích. Thiếu cấu hình ghi `BACKEND_NOT_CONFIGURED`, thiếu file ghi `PREREQUISITES_MISSING`.

### TRUST

Cài riêng theo <https://github.com/HavenFeng/TRUST>. Đặt `trust.enabled=true`, `trust.python`, `trust.repo` trong profile riêng. Có danh sách weights/FLAME/renderer bắt buộc trong profile; worker từ chối trước model init nếu thiếu weights để tránh vô tình chạy encoder ngẫu nhiên.

TRUST dùng đúng ảnh scene gốc của từng bộ, không dùng scene của 05 cho bộ khác. Đây là phép so sánh albedo chưa hiệu chuẩn; không khẳng định màu sắc thật từ ảnh ngược sáng.

## 6. In 3D

STL xuất đơn vị mm, đầu/cổ 22mm theo cấu hình demo. Chưa phải mô hình toàn thân 80–100mm hoàn chỉnh. Giữ `ready_for_machine=false` cho đến khi có bằng chứng quy trình chế tạo tương ứng.

Có hook gọi slicer, xuất machine file và UVtools trong `manufacturing`. Dùng **argv array**, không dùng một chuỗi shell. Placeholder: `{input_stl}`, `{output_dir}`, `{layers_dir}`, `{machine_file}`. Tự cấu hình executable/profile đúng máy/resin; không có lệnh UVtools/Saturn giả định sẵn. Wrapper slicer cần xuất các PNG layer đánh số zero-pad cùng kích thước vào `{layers_dir}`. Machine file phải được wrapper ghi vào đúng `{machine_file}` nếu muốn ghi nhận hash.

Bản này phân tích raster layer thật: lớp trống, thay đổi kích thước, component không chồng pixel với lớp trước. Đây là cờ connectivity để xem xét; không chứng minh lực support, bẫy resin, thời gian phơi hay độ bền. UVtools exit 0 cũng không tự đổi thành “in chắc chắn thành công”. Nếu chưa cấu hình slicer, trả `SLICER_NOT_CONFIGURED` rõ ràng.

## 7. Chốt tiêu chí đánh giá

So sánh từng bộ theo bốn câu hỏi riêng: mặt có giống không; tóc/kính có hài hòa không; mesh có hợp lệ không; file/quy trình có qua kiểm chứng chế tạo không. Không dùng IoU tóc hay khoảng cách mí mắt để đại diện cả bốn.

Bản này cung cấp cơ chế đối chứng và thuật toán thử nghiệm thực thi, không tuyên bố đã giải quyết triệt để ảnh mắt bị che, ngược sáng hoặc mọi kiểu tóc. Đọc `HEADLAB26-METHODS.md` và `verification/head261-release.json` để biết chính xác phần đã chạy, phần chưa chạy.

## Điểm mới của 26.1

- Đồ thị điều khiển mặc định tối đa 6000 nút, cell 0.55 mm, tách lớp trong/viền/ngoài để không gộp hai phía vỏ tóc. Mesh xuất giữ thứ tự đỉnh và toàn bộ faces gốc.
- Giữ vùng tiếp xúc, viền mỏng và nếp gấp sắc trên bản radial tham chiếu; làm mượt residual ARAP rồi backtracking khi có đảo hướng tam giác. Đây là hybrid radial + ARAP, không phải ARAP thuần bảo toàn phom hoặc thể tích.
- Replay05 giữ lại 12.5% residual sau smoothing; độ khác radial ở percentile95 khoảng 0.055 mm. Vì vậy chưa có bằng chứng thay đổi thẩm mỹ lớn, dù nhánh nhiều đỉnh đã chạy/xuất được.
- Trang so sánh có bộ lọc dataset/backend chạy offline. CSV/JSON vẫn chứa toàn bộ kết quả, không bị lọc mất dữ liệu.
- Config có `arap_cell_mm`, `arap_max_nodes`, `arap_min_fraction`; giảm `min_fraction` có thể cho phép kết quả gần radial hơn, không đồng nghĩa đẹp hơn.
