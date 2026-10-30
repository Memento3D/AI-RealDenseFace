# Update V3.2 — chân tóc từ ảnh đã detect

Giải nén ZIP code TRỰC TIẾP vào:
D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction

Cho phép thay thế các file code cùng tên. Không tạo thư mục con Memento3D-Figurine-V32. Không ghi đè runs hay dữ liệu cá nhân. Đây là bản update cho project V3/V3.1 đang có; không phải project cài mới độc lập.

Trong PowerShell tại thư mục project:
```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-FIGURINE-V32.ps1
```

Launcher lấy project từ vị trí của chính file PS1, dùng .venv hiện tại và tạo runs/figurine_v32_<thời gian> mới. Mặc định không cài lại dependencies, không tải model. Nếu máy chưa từng cài gói CPU V3 thì thêm -InstallDependencies. Không đụng .venv-rdf hoặc WSL conda.

Workspace và parser lấy từ configs/figurine04.v32.json; vẫn dùng đúng workspace case04 trước và parser D:/AI-cache/face-parsing/resnet18.onnx. Có thể truyền -Workspace 'đường_dẫn_head.workspace.html'. Checksum ảnh front phải khớp profile; nếu đổi người/ảnh cần profile riêng có checksum tương ứng.

## Thay đổi
- Không dùng mép tóc hình mũ tham số, giới hạn góc tóc mai/gáy, hay sóng trang trí làm chân tóc.
- Phân loại bề mặt từ mask và độ thấy được ở ba ảnh, trích đường biên rồi làm khối tóc bám đầu. Có thể giữ tóc mai, hõm quanh tai và gáy theo dữ liệu quan sát.
- Vùng khuất: nội suy giữa các nhãn đã quan sát; tô cam riêng và báo tỷ lệ. Không có ảnh phía sau nên không thể gọi toàn bộ tóc gáy là đã detect.
- Bề mặt bịt miệng/cổ nhân tạo không được sinh tóc. Các đảo nội suy không có neo tóc quan sát bị loại.
- Ghép mặt ngoài tóc và phần da còn lại thành đầu kín; giữ mô hình mặt gốc, không voxel hóa khuôn mặt. Đơn giản hóa phần đồng phẳng với dung sai 0.0001 mm.
- Mắt giữ cải tiến V3.1, không thêm thay đổi ánh nhìn. Tắt cười mặc định để đánh giá tóc riêng.
- Tắt gờ/rãnh/lọn trang trí. clumps/part/sweep/voxel_mm còn trong profile cho tương thích, không điều khiển đường tóc V3.2. base_mm, anchor_mm, edge_taper_mm và mask_dilation_px vẫn dùng. Không tăng dilation để chữa thiếu tóc gáy.

## Xem gì trước
1. Demo ZIP: mở compare-v31-v32.html để so trước/sau; chưa cần chạy lại.
2. review.html: mask xanh, biên mask cyan, biên dựng có hỗ trợ đỏ, đoạn nội suy cam.
3. hair_observation_support.glb/png: vùng tóc thiếu quan sát tô cam. Lớp cam có thể lộ qua khe miệng của preview gốc; STL đã đóng biên.
4. source_expression.print_mm.stl: đầu liền, đơn vị mm, trục Z lên. GLB: mét, Y lên.

hair_design_mm.stl chỉ phục vụ kiểm tra thiết kế tóc; KHÔNG phải tóc lắp rời đã thiết kế khe lắp/độ dày. Tóc bám đặc vào đầu; mép chuyển tiếp có offset 0.10 mm nên không suy ra độ dày vách in từ base_mm=0.6. Không kiểm chứng độ dày tối thiểu.

## Giới hạn
Đường biên phụ thuộc mask và camera đã khôi phục, không phải chuẩn độc lập. Mép tóc vẫn có thể cần làm mượt có kiểm soát sau khi duyệt mask. Đầu/biểu cảm gốc chưa được sửa. Chưa kiểm tra slicer/support/resin/in thật; không coi watertight là bảo đảm in được. Tỷ lệ đầu+cổ 35 mm là tỷ lệ thử. Resin không tự tạo màu preview.

Không train lại, không tạo từng sợi. Kiểm tra bản đồ chân tóc trước rồi mới thêm vài lọn lớn theo hướng tóc được xác nhận.
