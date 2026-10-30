# HeadLab V6 — kiểm tra và tự sửa vùng chân tóc

Đây là bản tích hợp thử nghiệm cho tóc ngắn bám đầu trên workspace hiện có.
Không phải hệ thống tái dựng mọi kiểu tóc và chưa xuất file máy in Saturn.

## Chạy đúng thư mục của bạn

Giải nén toàn bộ ZIP cập nhật **trực tiếp** vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Giữ ảnh sau tại `input\04\back.jpg`. Không cần di chuyển sang thư mục demo.
Giữ nguyên `.venv` hiện tại. Không thay `.venv-rdf` hoặc môi trường WSL/GPU.

```powershell
Set-Location "D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction"
powershell -ExecutionPolicy Bypass -File ".\RUN-HEADLAB-V6.ps1" -BackImage ".\input\04\back.jpg"
```

V6 dùng cùng dependencies CPU với V5; không cần cài lại nếu V5 đã chạy.
Nếu môi trường mới thiếu dependencies, thêm `-InstallDependencies`.
Profile mặc định ghim checksum bộ ảnh của ca 04; không tự bỏ kiểm tra này.

Mỗi lần chạy tạo thư mục mới `runs\headlab_v6_<timestamp>`, mở `compare.html`.
V5 và kết quả cũ được giữ lại. Các ứng viên thất bại vẫn có traceback riêng.

## Có gì trong một lần chạy?

1. Đọc ba ảnh trong workspace và ảnh sau; phân đoạn một lần, ước lượng camera như V5.
2. Dựng bản đối chứng V5 bốn góc nhìn.
3. Phân loại biên tóc–da, tóc–nền, biên mơ hồ; biên ảnh sau không được gọi là chân tóc đã đo.
4. Kiểm tra chiều sâu và hướng bề mặt. Giảm bằng chứng chân tóc ở mép chiếu của đầu,
   vì tóc nhìn thấy tại đó có thể thuộc phía xa.
5. Thử chín nghiệm có giới hạn, đo lại bốn ảnh. Giữ nghiệm giảm mục tiêu với các ngưỡng
   chống suy giảm từng góc nhìn và vùng dưới. Không ép đối xứng hoặc chiều dài gáy.
6. Chạy năm phương án thể tích (baseline, linear/robust × Laplacian/ARAP) cho cả
   hai nhóm, ở đầu+cổ 18 và 22 mm: 20 ứng viên, mỗi thuật toán lặp ba lần.
7. Kiểm tra lại từng cặp sau tối ưu thể tích. Nếu bản sửa kém hơn ngưỡng, tự quay
   về bản đối chứng cùng phương án. Chọn đầu ra nghiên cứu cho mỗi kích thước.
8. Xuất support thử nghiệm cho chính đầu ra được chọn, cùng báo cáo hình học.

## Nơi xem kết quả

- `compare.html`: mô hình trước/sau, các góc nhìn và toàn bộ ứng viên.
- `repair/repair.json`: chín lần thử, lần được giữ/bỏ và các số đo.
- `repair/*.edge_types.png`: xanh lam = tiếp giáp tóc–da; vàng = đường bao;
  hồng = biên mơ hồ. Những lớp này vẫn phụ thuộc chất lượng phân đoạn.
- `repair/repair_map.glb`: đỏ = bỏ tóc, xanh lá = thêm tóc, xám = phần còn lại.
  Đây là bản đồ chẩn đoán, không đi vào STL in.
- `selected/head_18p0mm/` và `selected/head_22p0mm/`: STL/GLB được chọn tự động,
  `selection.json` giải thích lựa chọn và các điểm chưa giải quyết.
- `selected/.../experimental_supports/`: thử nghiệm support và kiểm tra lớp voxel;
  không phải lớp cắt ở thông số máy/resin thật.
- `batch.json`, `metrics.csv`, `checksums.json`: đo lường và truy vết lần chạy.

## Chạy lại mà không nhận diện ảnh lại

Lần đầu V6 lưu `prepared/` gồm ảnh, nhãn, camera, hình học, texture và checksum.
Lần sau có thể dùng snapshot V6 này, không cần ONNX inference hay bake texture lại:

```powershell
powershell -ExecutionPolicy Bypass -File ".\RUN-HEADLAB-V6.ps1" `
  -PreparedRun ".\runs\headlab_v6_THAY_BANG_LAN_CHAY_THAT\prepared"
```

Đường dẫn trên là ví dụ cần thay bằng thư mục V6 thực tế.
**Không trỏ tới kết quả V5:** V5 chưa lưu đủ nhãn semantic để replay chính xác.
Cache phải khớp profile và toàn bộ checksum; tuyệt đối không dùng pickle.

## Đọc kết quả đúng phạm vi

Các ngưỡng hiện là ngưỡng thử nghiệm, chưa được hiệu chuẩn trên tập nhiều người.
Điểm số là độ nhất quán với mask dùng để tối ưu, không phải nhận dạng cá nhân độc lập.
Đầu ra có thể tự quay về V5 nếu chưa tìm được sửa tốt hơn; đó là hành vi dự kiến.

Camera ảnh sau vẫn dùng mốc tai gần đúng, tiêu cự giả định và ưu tiên pose. Nếu biến
độ cao tóc chạm cận hoặc độ khớp thấp, `selection.json` báo giới hạn đầu vào/biểu diễn.
Không biến việc khớp vài mốc rất tốt thành tuyên bố camera đúng.

V6 không thay tọa độ khuôn mặt nguồn để làm đẹp, không train lại model, không phát
minh chân tóc bị che. Cách biểu diễn hiện vẫn cần một nhánh khối tóc rời da đầu cho
tóc dài, búi hoặc buộc; không nên áp profile tóc ngắn này cho mọi người.

`ready_for_machine` vẫn là `false`: mesh kín và support voxel không chứng minh
độ bền, độ dày tối thiểu, peel force, lớp cắt hoặc thông số resin. Mục tiêu cuối
tượng 80–100 mm có thân và đế chưa được tạo trong batch đầu này.
