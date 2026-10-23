# v0.13.1 · Sửa mask làm thủng mắt, mũi và môi

Lỗi ở v0.13: mask dựa vào xác suất lớp thắng cao nhất. Tại biên skin/nose/lip/eye/brow, hai lớp đều cần giữ nhưng mỗi lớp có thể chỉ có xác suất xấp xỉ 0.5. Ngưỡng 0.55 loại nhầm pixel biên, phép co mask và loại cả tam giác làm khe lớn hơn trên đầu 3D.

Bản sửa giữ điều kiện nhãn thuộc tập lớp cần dùng, nhưng ngưỡng tin cậy áp lên **tổng xác suất tất cả lớp cần giữ**. Không fill mù qua tóc/áo/nền, không hạ ngưỡng toàn cục. Màu nền được thêm ánh sáng nhẹ để bớt phẳng; màu ảnh vẫn không chiếu sáng thêm.

## Chạy lại

Cập nhật nội dung source ZIP vào dự án. Không cài lại môi trường hoặc tải lại model. Chạy:

```powershell
.\.venv\Scripts\python.exe .\scripts\optimize_head.py `
    --head-run runs/rdf_open_20261023_152718_f3ce74/three_photo_7cf7011b `
    --texture-run runs/rdf_open_20261023_152718_f3ce74/three_photo_7cf7011b/three_textures_73087dd7 `
    --auto-mask `
    --face-parser-model D:/AI-cache/face-parsing/resnet18.onnx
```

Không truyền lại auto_masks.json cũ qua --masks-json, vì nó vẫn chứa các khe cũ. Mở workspace mới được terminal in ra. Source vẫn đầy đủ và gồm setup model cho người chưa dùng v0.13, xem UPDATE-v0.13.md.

## Đã kiểm chứng

- Chạy model ONNX CPU thật trên đúng ba ảnh gốc, xác nhận hash ảnh trùng nguồn bạn dùng.
- Dùng đúng crop trong report; mask v0.13 tái tạo giống hoàn toàn các PNG mask đã tải lên.
- So lại mask sửa: giữ liên tục vùng mắt, lông mày, mũi và môi; tóc/áo/nền vẫn được loại phần lớn.
- Pixel mask được giữ trước → sau: front 111197 → 118045; left 119131 → 126502; right 125975 → 139947. Đây chỉ là diện tích mask, không phải score chất lượng.
- 136 unit tests và architecture check đạt; thêm test trường hợp biên giữa hai lớp được giữ.

Chưa chạy lại toàn bộ đầu 3D thật vì chưa có viewer nguồn/mesh tương ứng trong môi trường kiểm tra này. Kết quả cần xem lại trong workspace trên máy bạn. Tai nghe vẫn có thể còn sót; model không có lớp tai nghe riêng. Report warnings rỗng không chứng minh mask chính xác. Không thay đổi model dựng đầu hoặc tự xác nhận likeness.
