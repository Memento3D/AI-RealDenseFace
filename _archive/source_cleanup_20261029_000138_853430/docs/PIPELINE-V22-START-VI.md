# Memento3D Pipeline V2.2 — da nền thống nhất, tóc có khối và tóc mai cục bộ

Bản này sửa hai vấn đề được phản hồi sau V2.1: bóng ảnh còn làm loang da, và tóc mai của mẫu thư viện quá cao/cụt. Giữ nguyên tọa độ mặt và mắt nguồn. Không training thêm mô hình AI. Đây là demo hình học và vật liệu thiết kế; chưa phải đầu ra đã được kiểm chứng sẵn sàng chạy máy resin.

## Chạy đúng thư mục Windows hiện tại

Giải nén **nội dung gói mã trực tiếp** vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Sau giải nén, `RUN-PIPELINE-V22.ps1` nằm ngay tại thư mục dự án, cùng cấp `.venv`, `scripts`, `configs`. Không tạo thêm lớp thư mục Memento3D-Figurine-V3 hoặc Pipeline-V22 ở ngoài.

Từ PowerShell tại thư mục đó:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-PIPELINE-V22.ps1
```

Launcher dùng `.venv\Scripts\python.exe` hiện có, kiểm tra mã/dependencies CPU và tạo lần chạy mới trong `runs\pipeline_v22_<thời gian>`. Không ghi đè kết quả cũ. Không thay môi trường WSL/CUDA/RDF. Nếu thiếu dependencies, cài bộ CPU vào venv chính. `-InstallDependencies` dùng khi cần ép kiểm tra/cài lại bộ dependency. ExecutionPolicy Bypass chỉ áp dụng cho tiến trình PowerShell này.

Mặc định dùng workspace đã chạy mẫu 04:

`runs/v1_20261025_132459_bbbc53c9/three_photo_346bb6bf/head_workspace_3cdeba95/head.workspace.html`

Parser: `D:/AI-cache/face-parsing/resnet18.onnx`.

Có thể chỉ định đường dẫn khác bằng `-Workspace` và `-ParserModel`. Gói không chứa parser hoặc workspace cá nhân. Có chứa mã đầy đủ và hai STL thư viện đã dùng; V2.2 hiện chỉ chấp nhận tóc ngắn 21778.

## Tăng tốc nếu đã có kết quả V2.1 hợp lệ

Không bắt buộc. Nếu còn thư mục V2.1 đã chạy đúng cùng workspace và cùng kích thước, có thể truyền:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-PIPELINE-V22.ps1 -ReferenceRun ".\runs\pipeline_v21_THOI_GIAN_DA_CHAY"
```

Thay `pipeline_v21_THOI_GIAN_DA_CHAY` bằng thư mục có thật, chứa `report.json`, `checksums.json` và ứng viên `compact_21778_0p25/appearance.glb`. Đây không phải yêu cầu chạy lại V2.1.

Mã kiểm tra workspace SHA, tỉ lệ, asset SHA, checksum GLB, thứ tự/hướng vertex rồi kiểm chứng 128 ray cavity với mẫu nguồn. Dữ liệu khác chủ thể, khác scale, mesh đã repair hoặc sai checksum bị từ chối, không dùng âm thầm. Không có `ReferenceRun`, mã tính correspondence mới. Kết quả thực tế trong gói Results04 dùng đường tăng tốc có kiểm chứng; thời gian đó không phải benchmark chạy mới hoàn toàn.

## Sửa da như thế nào

- Chọn vùng semantic skin sáng, đủ lớn và liên thông; bỏ điểm cháy sáng. RGB tham chiếu là quyết định thiết kế từ ảnh chưa hiệu chuẩn.
- Mọi vùng da nền của mặt/tai/cổ và phần bị che dùng cùng một màu. Không dùng bóng ảnh làm màu da.
- Chỉ đưa pixel mắt–mày–môi nguồn vào vùng feature tương ứng. Chuyển tiếp nằm bên trong feature; không phục hồi nguyên ô tam giác mang theo da tối như V2.1.
- Geometry của mặt và mắt nguồn không thay đổi. Không pose nụ cười hoặc vẽ lông mày theo một preset chung.
- GLB `appearance.glb` dùng chiếu sáng vật liệu PBR; `unlit.glb` dùng cùng màu nền nhưng tắt chiếu sáng trên đầu để kiểm tra base color. Mặt vẫn sáng/tối theo đèn khi xem bản PBR, đó là ánh sáng của cảnh.
- Màu là thiết kế albedo phẳng để có sản phẩm đồng nhất; không tuyên bố phục hồi chính xác albedo sinh học. Nếp da/vết bớt chưa được bảo toàn vì phiên bản này chủ động chuẩn hóa da nền.

## Sửa tóc như thế nào

1. Cavity của mẫu tóc được fit theo đầu nguồn.
2. Tách độ dày nền khỏi chi tiết radial của mẫu; làm mượt trường radial để lấy phần chi tiết còn lại. Không nén toàn bộ lọn theo cùng một hệ số như V2.1.
3. Tìm vùng tai từ semantic projections có hỗ trợ ảnh nguồn. Dựa vùng trước tai để chọn điểm điều khiển tóc mai.
4. Căn tóc mai theo mép trên tai; thêm mức custom có giới hạn. Kéo dài bằng bài toán dịch chuyển Laplacian cục bộ với bên ngoài vùng sửa cố định. Thuôn bề rộng, bo ngọn, rồi chiếu lại lên sọ để giữ khoảng cách thành tóc.
5. Kiểm tra vùng mặt phía trước, tiếp xúc theo ray mẫu, phần tai bị phủ thêm và topology sau union/STL round-trip. Chỉ riêng tóc được phép repair voxel có giới hạn nếu cần, không voxel hóa khuôn mặt.

Độ dài tóc mai là **custom thiết kế theo mốc tai**, không phải số đo chiều dài tóc cá nhân suy ra từ ảnh. Report ghi rõ khi không có đủ hair pixels quanh mốc tai. Không biến chiều dài custom thành “đã đo được”. V2.2 không triển khai ARAP; Laplacian ở đây là biến thể tuyến tính trên trường dịch chuyển, không phải toàn bộ thuật toán quay bất biến trong bài báo.

## Chỉnh nhẹ các tham số

Trong `configs/pipeline-v22-04.json`:

- `sideburns.preferred_extra_mm`: chọn mức custom đề xuất trong batch.
- `sideburns.extras_mm`: các mức chạy một lần để so sánh (mặc định 0 và 0.4 mm thêm sau bước căn theo tai).
- `sideburns.ear_top_clearance_mm`: vị trí mục tiêu tương đối với mép trên tai.
- `sideburns.tip_taper_factor`: độ thuôn bề rộng ngọn mai.
- `compact_fit.body_mm`: độ phồng nền, mặc định 0.52 mm.
- `compact_fit.detail_strength`: mức khôi phục chi tiết radial, mặc định 0.8.

Các giá trị là tham số tạo hình cho đầu/cổ 22 mm, không phải thông số resin đã hiệu chuẩn. Không tăng tùy tiện vượt giới hạn trong mã. Mẫu 21778 vẫn quy định kiểu mái và viền tóc chính; bản này chưa fit chân tóc cá nhân toàn bộ hoặc dựng tóc gáy từ ảnh sau.

## Xem kết quả

Giải nén gói Results04 riêng và mở `review.html` để xem trước mà không chạy lại.

- `selected/appearance.glb`: màu nền chung + chiếu sáng vật liệu.
- `selected/unlit.glb`: kiểm tra màu nền không chiếu sáng trên đầu.
- `selected/head_mm.stl`: khối đầu–tóc, mm và Z up.
- `selected/hair_mm.stl`: tóc riêng để đánh giá, chưa có cơ cấu tháo lắp/dung sai.
- `shared/bright_skin.json`: tham chiếu và kiểm tra sai lệch da nền.
- `*/candidate.json`: vị trí tóc mai trước/sau, số vertex sửa, contact/ear coverage, sai lệch mặt.
- `*/surface_audit.json`: kiểm tra cạnh và vertex-link sau đọc lại STL.
- `selected/experimental_supports`: support thử nghiệm, không phải lớp cắt máy đã xác minh.

Printer Saturn 4 Ultra 16K in resin một màu; màu GLB không tự xuất hiện trên STL hoặc bản in. Chưa có hiệu chuẩn resin, kiểm tra độ dày chính xác toàn cục, chứng minh không tự giao toàn bộ, mã máy/slicer hay in vật lý. `ready_for_machine=false` vẫn được giữ. Các kiểm tra tự động là cổng loại lỗi đo được, không phải điểm đẹp hoặc độ giống người thật đã kiểm chứng.

Launcher chưa được thực thi trên Windows trong môi trường kiểm thử này; Python CPU và dữ liệu mẫu đã được chạy thực tế, số liệu nằm trong `ACTUAL-RUN-VI.md` của gói kết quả.
