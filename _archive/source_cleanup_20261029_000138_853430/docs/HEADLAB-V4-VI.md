# HeadLab V4 — một lệnh, một bộ so sánh thuật toán

Đây là bản UPDATE cho project Memento3D-Head-Reconstruction đang có. Giải nén nội dung ZIP trực tiếp vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Không tạo thêm thư mục con Memento3D-HeadLab-V4. Cho phép thay thế những file code cùng tên. Không thay dữ liệu, runs, môi trường WSL hoặc .venv-rdf.

## Chạy

PowerShell tại project:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB-V4.ps1
```

Dùng `.venv` hiện tại; mặc định không pip install và không tải model. Nếu thiếu thư viện:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB-V4.ps1 -InstallDependencies
```

Mặc định 5 phương án × 2 kích thước đầu+cổ (18 và 22 mm) × 3 lần giải = 30 phép chạy thuật toán, xuất 10 ứng viên. Mỗi ứng viên chỉ render/xuất một lần. 18/22 mm là hai giả thuyết nghiên cứu phục vụ tượng tổng 80–100 mm, KHÔNG phải tỷ lệ cơ thể đã chốt. Chưa dựng thân hoặc đế.

Để thử nhanh tại một kích thước, vẫn so cả 5 phương án:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB-V4.ps1 -Repeats 1 -HeightsMm 20
```

Hoặc chạy CLI Python trực tiếp với `--profile`, `--workspace`, `--parser-model`, `--output`, `--repeats` và `--heights-mm`.

## Đầu vào

Dùng đúng workspace case04 và parser trong `configs/headlab04.v4.json`. Parser phải khớp SHA256 đã pin của project. Đổi người/ảnh cần study profile có checksum ảnh front tương ứng. Không bỏ checksum chỉ để ép một ảnh khác chạy qua.

Bản này bắt đầu từ head.workspace.html đã dựng, sử dụng ba ảnh nhúng trong workspace. Đây KHÔNG phải một model tái dựng đầu mới từ ảnh thô. Camera, mask và đầu gốc dùng chung cho mọi ứng viên. Chỉ parse ba ảnh một lần và bake màu một lần cho cả batch; dữ liệu chia sẻ lưu tại shared/.

## Năm phương án thực sự chạy

1. baseline: chân tóc V3.2, độ dày tóc nền có taper; không fitting hoặc biến dạng thêm.
2. linear_laplacian: bounded TRF với linear loss + uniform Laplacian displacement regularization.
3. robust_laplacian: cùng dữ liệu/TRF nhưng soft_l1 loss + cùng Laplacian.
4. linear_arap: linear loss + local/global ARAP với trọng số cạnh dương, đồng đều.
5. robust_arap: soft_l1 loss + cùng ARAP.

2×2 là so sánh factorial có kiểm soát. Fitting dùng cùng tham số ban đầu, cùng điểm lấy mẫu, bounds và budget. Baseline không có chi phí fitting nên không coi tốc độ baseline là tốc độ của solver. Các lần lặp là cùng một bài toán tất định để đo thời gian, không phải thêm người hay thêm quan sát độc lập.

Laplacian ở đây là bản regularization của nhóm theo nguyên tắc Laplacian, KHÔNG phải triển khai đầy đủ LSE gốc với bất biến quay. ARAP là spokes local/global tự triển khai theo nguyên tắc bài báo; không tuyên bố là phần mềm gốc hoặc có tốc độ như bài báo.

Các bộ biến dạng hiện áp vào khối tóc đã fitting; không tự làm thay đổi mặt để tạo khác biệt. Mặt và mí gốc trong appearance GLB được giữ nguyên. Boolean mắt/tóc làm thay đổi mặt ngoài của STL ở các vùng giao nhau. Đây chưa phải benchmark chỉnh mặt theo tải cơ học.

## Cách khớp tóc và giới hạn

Khối tóc có 10 tham số độ nâng dùng cơ sở RBF, bám theo bề mặt đầu. Điểm gần tiếp tuyến nhìn thấy được chiếu vào signed distance của mask để khớp đường bao gần đúng. Có prior giới hạn độ nâng và taper giữ chân tóc. Không phải differentiable renderer, không phải mô phỏng tóc vật lý, không suy ra hướng sợi.

Chân tóc quan sát dùng nhãn mask. Vùng khuất kế thừa nội suy harmonic; không gọi đó là detect đúng phía sau. Cấu trúc hiện hỗ trợ tóc ngắn bám đầu. Tóc dài, điểm buộc, búi và nơ CHƯA triển khai. Không tuyên bố đã có mọi kiểu tóc chỉ vì có utility đường cong.

Centripetal Catmull–Rom có utility và unit tests nhưng chưa dùng trong 5 ứng viên. Hair Meshes, CGAL SDF, mô phỏng sức bền và UVtools được ghi rõ là nguồn tham khảo, KHÔNG ghi là đã chạy.

## Đọc kết quả

- `compare.html`: toàn bộ ứng viên, ảnh xám trước/nghiêng/sau, ảnh màu, chồng mask và đường bao.
- `metrics.csv`: mở bằng Excel; cùng cấu trúc cho cả ứng viên lỗi.
- `batch.json`: thông tin nguồn, cấu hình, thời gian cả batch và từng ứng viên.
- `<ứng_viên>/candidate.json`: solver, tham số, objective ARAP, độ dịch chuyển, mesh checks.
- `<ứng_viên>/head_mm.stl`: hình học đầu+cổ, mm, Z lên. KHÔNG có support.
- `<ứng_viên>/appearance.glb`: mét, Y lên, màu là hướng dẫn sơn.
- `sources.json`: nguồn gốc và mức triển khai của từng phương pháp.
- `checksums.json`: checksum toàn bộ đầu ra.

Trong từng kích thước, ảnh dùng cùng framing và ánh sáng để dễ so. Xanh là mask, tím là biên tóc dự đoán. IoU chỉ đo nhất quán với ảnh dùng fitting và camera khôi phục, KHÔNG phải độ giống người hay chuẩn camera độc lập. Không tự chọn người thắng từ IoU.

Thời gian algorithm chỉ gồm fit + deform, không gồm parse, bake, Boolean/export, kiểm tra mesh hoặc render. Các thời gian này ghi riêng; end_to_end_seconds ghi toàn batch. RAM nếu không đọc được trên môi trường hiện tại thì để null, không coi thiếu số đo là lỗi thuật toán. Không đo GPU.

## Kiểm tra chế tạo và trạng thái trung thực

Kiểm tra tự động mesh kín, winding, thành phần, hữu hạn, thể tích. Bổ sung lấy mẫu ray hướng vào trong để tìm các đoạn xuyên khối ngắn và tìm hướng đặt bằng proxy diện tích mặt hướng xuống. Đây là SCREENING, không phải minimum wall thickness chính xác, không phải CGAL SDF, không phải FEA và không chứng minh độ bền.

`trial_feature_mm = 0.4` chỉ là ngưỡng thử để so sánh; không phải đặc tính đã đo của Saturn hoặc resin. Không báo `ready_for_machine=true` từ mesh kín hoặc ray samples. Mặc định profile resin, layer height và exposure chưa biết; support/slicing driver chưa cấu hình, nên trạng thái in luôn chưa xác minh.

Có adapter tùy chọn để gọi executable support/slicing bên ngoài bằng argv (không shell), chờ có timeout và kiểm tra hash đầu vào/profile/file máy cùng báo cáo kiểm tra. Không có backend giả. Adapter KHÔNG tự cài hay giả lập Chitubox/UVtools. Ngay cả adapter trả thành công vẫn bị chặn nếu kiểm tra giao cắt và độ dày chính xác chưa có; hiện hai cờ này chưa được triển khai nên chưa thể đạt đầy đủ gate.

Nếu một ứng viên lỗi, các ứng viên khác vẫn chạy. Trang so sánh giữ lỗi và batch trả exit code 2. Không ghi đè batch cũ.

## Cách dùng để quyết định thuật toán

Đánh giá trong cùng kích thước: khối tóc/chân tóc, mức dịch chuyển, hội tụ, hình học, thời gian. Chưa thay weights sau khi xem kết quả rồi gọi đó là test độc lập. Sau khi chọn cấu hình cần thử trên người khác và có ảnh giữ lại, rồi kiểm chứng với slicer/in thật. Nghiên cứu có người đánh giá kết quả không có nghĩa vận hành cần sửa mesh thủ công.

Bản này giải quyết việc chạy và so sánh cùng một lần. Chưa phải hệ thống ảnh thô → file chạy máy hoàn chỉnh, chưa có in vật lý xác nhận. Bước tích hợp profile resin và slicing vẫn là điều kiện bắt buộc trước mục tiêu tự động sản xuất.
