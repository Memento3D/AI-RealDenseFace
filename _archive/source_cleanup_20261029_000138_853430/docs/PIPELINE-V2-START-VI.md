# Memento3D Pipeline V2.0 — mẫu tóc, mặt cá nhân hóa và kiểm tra chế tạo

## Cài đúng vị trí hiện tại

Giải nén ZIP cập nhật **trực tiếp** vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

`RUN-PIPELINE-V2.ps1` phải nằm cùng cấp với launcher V5/V6 cũ. Gói đã gồm hai
STL demo và thư viện `assets/hair/library.json`. Không cần ảnh back cho nhánh V2.
V2 vẫn cần workspace đầu nguồn hiện có và parser đã dùng ở V5/V6.

```powershell
Set-Location "D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction"
powershell -ExecutionPolicy Bypass -File ".\RUN-PIPELINE-V2.ps1"
```

Launcher chỉ sử dụng `.venv\Scripts\python.exe`; không thay môi trường RDF/WSL/GPU.
Nếu thiếu dependencies, tự cài bộ CPU vào `.venv` chính. Dependency mới đáng chú ý
là `rtree`, phục vụ truy vấn giao tia/bề mặt. Không tải model mới hoặc train model.

Mỗi lần chạy tạo một thư mục riêng `runs\pipeline_v2_<timestamp>` rồi mở
`review.html`. Mặc định chạy **5 ứng viên**, không chạy ma trận HeadLab cũ:

- Hai mẫu tóc × hai biểu cảm: source và soft_smile, khớp bằng radial cage.
- Một đối chứng affine cho mẫu đầu tiên và biểu cảm đầu tiên.

## Chạy nhanh hoặc custom

Một mẫu ngắn, một biểu cảm nhẹ, không đối chứng affine:

```powershell
powershell -ExecutionPolicy Bypass -File ".\RUN-PIPELINE-V2.ps1" -Quick
```

Chọn mẫu tóc, màu da, kích thước đầu+cổ và mức custom:

```powershell
powershell -ExecutionPolicy Bypass -File ".\RUN-PIPELINE-V2.ps1" `
  -HairIds "21778" -Expression "soft_smile" -SkinId "3" `
  -HeadMm 22 -Volume 0.05 -FringeLiftMm 0.2 -NapeLiftMm 0.1
```

- `21778`: mẫu ngắn dạng cụm; `11256`: mẫu tóc lượn rộng hơn.
- `source`: giữ tọa độ đầu nguồn trước bước hợp khối.
- `soft_smile`: preset nâng khóe môi và má nhẹ; không phải biểu cảm tái dựng chính xác.
- `SkinId`: 1–8 theo bảng màu dự án trong profile. Đây **không phải thang Monk**.
- `Volume`: -0.20..0.20; `FringeLiftMm`/`NapeLiftMm`: -1..1 mm;
  `HairYawDeg`: -15..15 độ. Đạt giới hạn tham số chưa có nghĩa kết quả chắc chắn
  đạt hình học; mọi cấu hình vẫn kiểm tra lại.
- `Teeth`: auto/hidden/show. Mặc định auto chỉ đề xuất màu răng nếu vùng miệng có
  đủ pixel sáng ít bão hòa; không dựng răng rời hoặc ép ảnh khép môi thành cười hở răng.

Đường dẫn khác có thể truyền bằng `-Workspace` và `-ParserModel`.
Không thay đổi kích thước đầu tùy tiện sau khi kiểm định để đưa đi in.

## Kết quả và cách đọc

| Đường dẫn | Nội dung |
|---|---|
| `review.html` | So sánh tóc/biểu cảm, lọc ứng viên, số đo và bằng chứng |
| `selected/head_mm.stl` | Khối đầu–tóc được đề xuất; mm, Z-up |
| `selected/appearance.glb` | Màu nền và đặc trưng mắt–môi–mày; meter, Y-up |
| `selected/experimental_supports/` | Hướng in/support thử nghiệm và báo cáo voxel |
| `asset_audits/*.json` | Hai STL nguồn: cạnh, liên kết đỉnh, Euler, kích thước, tia độ dày lấy mẫu |
| `hair_*/fitted_hair_audit.json` | Tóc sau khi khớp và sửa theo từng ứng viên |
| `hair_*/final_surface_audit.json` | Kiểm tra bề mặt của STL cuối đã xuất và đọc lại |
| `evaluation.json` | Tách A tóc, B mặt, C chế tạo; đo được/proxy/chưa kiểm chứng |
| `input_assessment.json` | Thống kê ảnh/semantic và độ khớp camera nội bộ |
| `human_review.schema.json` | Mẫu thu thập nghiên cứu; chưa có điểm chuyên gia giả định |
| `comparison.csv`, `report.json`, `checksums.json` | So sánh và truy vết |

“DEMO_EXPORTED” nghĩa batch đã xuất ứng viên qua các kiểm tra hình học được cài
đặt. Không tương đương “sẵn sàng sản xuất”. `ready_for_machine` vẫn false khi
chưa hiệu chuẩn resin, chưa chạy slicer thực và chưa kiểm chứng in vật lý.

## Phạm vi thực tế

V2 này bắt đầu từ **head.workspace.html đã tái dựng**, không thay thế pipeline
ảnh → đầu nguồn ở phía trước. Thư viện tóc thay được trong phần mềm; bản STL
đầu tiên là khối hợp nhất, chưa có ngàm tháo lắp tóc vật lý.

Tóc có thể được sửa voxel riêng khi gặp đỉnh bị thắt/pinch hoặc lỗi STL số học;
khuôn mặt không được voxel-remesh. Mọi lần sửa có giới hạn, lưu thay đổi thể tích
và sai lệch bề mặt lấy mẫu, rồi kiểm tra lại. Nếu hết ngân sách sửa, loại ứng viên.

Màu là vật liệu xem trước/hướng dẫn hoàn thiện. Máy Saturn resin hiện tại không
in được các màu mắt, môi, tóc từ GLB. Tách tóc không tự tạo khả năng in nhiều màu.

Đây là batch phần đầu+cổ 22 mm mặc định, cho mục tiêu tượng tổng thể 80–100 mm.
Chưa tạo thân và đế tượng. Đế/raft support không phải đế sản phẩm.

## Thêm mẫu tóc sau này

Đọc `PIPELINE-V2-ASSET-CONTRACT.md`. Không cần sửa thuật toán cho mỗi mẫu, nhưng
cần chuẩn hóa hướng, đơn vị và metadata vùng lòng tóc một lần. Dùng importer
ở chế độ kiểm tra trước, rồi `--register` để thêm ID mới; không ghi đè ID cũ.
Profile và registry là dữ liệu có thể chỉnh; mã nguồn có checksum riêng.

## Tài liệu đi kèm

- `PIPELINE-V2-EVALUATION-VI.md`: năm góc nhìn, tiêu chí A/B/C, Q&A và lộ trình.
- `PIPELINE-V2-METHODS.md`: công thức thực hiện, tham số thử nghiệm và nguồn.
- `PIPELINE-V2-ASSET-CONTRACT.md`: giao diện thư viện tóc và điều kiện thêm mẫu.
