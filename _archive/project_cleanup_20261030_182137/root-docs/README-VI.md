# Hair on source head — khắc phục mảng tóc phẳng của benchmark

Bản cập nhật này là **bước dựng/ghép đầu ra hoàn chỉnh sau benchmark**, không phải chỉ đảo normals và không chạy lại một model gen tóc khác. Dùng đầu 3D gốc làm khung, fit chân tóc từ các ảnh nguồn, tạo khối tóc quanh sọ rồi union thành STL đầu–tóc. Native TripoSR chỉ làm relief trước có giới hạn, không dùng nguyên mảng crop làm bộ tóc.

## Cài vào đúng dự án

Giải nén nội dung gói mã trực tiếp vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Phải thấy `RUN-HAIR-ON-HEAD.ps1`, `OPEN-HAIR-ON-HEAD.ps1` ở ngay thư mục dự án; mã nằm trong `hair-on-head`. Không đổi chỗ folder `triposr`, không cần chép đè native và không dùng lại bản native đã sửa normals như một bộ tóc hoàn chỉnh.

## Chạy từ thư mục bạn đang đứng

Bạn đang ở:

`...\runs\hair_bench_20261028_192116_818`

Chạy:

```powershell
powershell -ExecutionPolicy Bypass -File ..\..\RUN-HAIR-ON-HEAD.ps1 -BenchRun .
```

Hoặc từ thư mục gốc dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-ON-HEAD.ps1
```

Launcher chọn benchmark gần nhất nếu không chỉ định, tìm workspace đúng SHA và dùng `.venv\Scripts\python.exe` chính. Tạo thư mục **`fitted_hair_<thời gian>` bên trong benchmark hiện tại**, không ghi đè file cũ. Tự mở trình xem 3D mới; tóc và đầu đã cùng hệ tọa độ, không có bước kéo/chỉnh cỡ thủ công.

Bước này dùng CPU, không tải model gen mới, không chạy lại TripoSR/Hunyuan, không thay môi trường CUDA/WSL. Nếu thiếu thư viện CPU, launcher cài bộ trong `hair-on-head/requirements-cpu.txt` vào venv chính.

## Các đầu vào được dùng

- `subject/subject.json` và `subject/original_face.glb` trong benchmark: kiểm tra SHA để giữ đúng người/đúng mặt.
- Workspace gốc cùng SHA: tự tìm `hair-bench/input04/head.workspace.html`, sau đó đường dẫn `runs/v1_20261025_132459_bbbc53c9/three_photo_346bb6bf/head_workspace_3cdeba95/head.workspace.html`.
- Parser đang có: `D:\AI-cache\face-parsing\resnet18.onnx`.
- `triposr/native.glb`: tùy chọn, chỉ lấy relief trước. Không có native vẫn dựng khung tóc trên sọ được.
- `input/04/back.jpg`: tự nhận nếu tồn tại; chỉ dùng khi có vùng tóc và hai tai đủ hỗ trợ căn ảnh. Ảnh khác trùng tên hoặc không đủ vùng đầu bị từ chối, không đoán camera. Bạn có thể truyền `-BackImage` để đổi đường dẫn.

Nếu cần chỉ định workspace/parser khác:

```powershell
powershell -ExecutionPolicy Bypass -File ..\..\RUN-HAIR-ON-HEAD.ps1 -BenchRun . -Workspace "D:\duong-dan\head.workspace.html" -ParserModel "D:\AI-cache\face-parsing\resnet18.onnx"
```

`-NoBackImage` cho phép chạy rõ ràng với ba ảnh của workspace: phần gáy vẫn có khối theo sọ, nhưng chân tóc gáy là prior có giới hạn, chưa phải quan sát cá nhân. Không dùng chế độ này để tuyên bố đã tái dựng đúng gáy. Lần kiểm thử đính kèm dùng ba ảnh; không có đúng ảnh sau gáy hợp lệ trong bộ file kiểm thử. Nhánh nhận ảnh gáy có kiểm tra từ chối đầu vào sai, chưa được xác nhận trên ảnh gáy của bạn trong lần chạy này.

## Xem lại mà không chạy lại

Từ thư mục benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File ..\..\OPEN-HAIR-ON-HEAD.ps1 -BenchRun .
```

Hoặc truyền `-FitRun` tới thư mục `fitted_hair_...` đã có. Server xem chỉ nghe trên `127.0.0.1`, tự hết hạn sau một giờ. Mở lại không gen/infer hay fit lại.

Không đưa `hair_on_head.glb` vào luồng tự chuẩn hóa bounding box của Studio cũ: file mới đã ở đúng vị trí của đầu, chuẩn hóa thêm lần nữa sẽ làm sai. Dùng cửa sổ mới launcher mở hoặc nạp **`head_with_hair.glb`** trực tiếp vào Blender để xem cả bộ.

## Đầu ra

- `head_with_hair.glb`: đầu và tóc đã ghép, cùng tọa độ và scale. Buffer vị trí mặt/mắt và ảnh texture mặt được so byte/hash, không đổi nét hoặc tô lại da trong bản sửa tóc này.
- `hair_on_head.glb`: tóc riêng, **vẫn ở tọa độ đầu**, đơn vị GLB meter.
- `head_mm.stl`: khối đầu–tóc sau union, mm và Z up. Đầu/cổ nguồn cao 22 mm theo profile.
- `hair_mm.stl`: tóc riêng để đánh giá hình học; chưa phải tóc tháo lắp có dung sai cơ khí.
- `geometry.png`, `appearance.png`, `view_*.fit.png`: ảnh kiểm tra thực tế.
- `report.json`, `review.html`: bằng chứng, cổng kiểm tra và phần chưa quan sát.
- `experimental_supports`: support thử nghiệm được tính lại cho khối mới, không phải mã máy hoặc lớp cắt thực đã chứng minh.

## Tại sao hết lỗi “mảng phẳng không vừa đầu”

1. Không yêu cầu TripoSR đoán toàn bộ bộ tóc từ một ảnh crop trống vùng mặt/sọ.
2. Khung tóc lấy bán kính trực tiếp từ đầu nguồn ở mọi hướng, gồm hai bên và sau.
3. Đường chân tóc được fit bằng mask có kiểm tra thấy/che khuất và góc nhìn. Ảnh chính diện/nghiêng không sở hữu vùng chân tóc sau gáy trung tâm.
4. Độ phồng là trường ít tham số có giới hạn; phần cực đỉnh không phụ thuộc góc quanh trục, tránh tạo chóp/đường nối ở đỉnh.
5. Detail native được lọc bỏ phồng nền, giới hạn 0.14 mm và tắt gần đường chân tóc/cực đỉnh. Không mang phần mảng phẳng hoặc gai dài native vào kết quả.
6. Kiểm tra giao với đầu, vùng mặt bảo vệ, tám ray phủ trước/hai bên/sau, hướng mặt, thành phần và vertex-link sau STL round-trip.

## Giới hạn cần đọc đúng

Đây là chế độ tóc ngắn bám sọ, không phải giải pháp cho tóc dài, búi/bím hoặc tóc rời khỏi đầu. Gáy chưa có ảnh được ghi là chưa quan sát; camera phục hồi từ workspace không phải hiệu chuẩn độc lập. IoU là độ khớp mask nguồn, không phải điểm đẹp/giống người đã được hội đồng chấm.

STL kín không đủ để xác nhận in được trên mọi resin/máy. Chưa đo độ dày chính xác toàn bộ, chứng minh không tự giao toàn cục, hiệu chuẩn resin, kiểm tra slicer/mã máy hoặc in vật lý. `ready_for_machine=false` vẫn được giữ. Màu GLB không tự xuất hiện trên bản in resin một màu.

PowerShell launcher chưa được thực thi trên Windows trong môi trường kiểm thử này. Python CPU, các phép đo mesh, ảnh render và HTTP phục vụ viewer đã chạy thực tế. WebGL tương tác chưa được kiểm thử tự động tại đây.
