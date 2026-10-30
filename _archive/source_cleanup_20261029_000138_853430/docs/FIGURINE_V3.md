# Figurine V3 — đầu cá nhân hóa, ưu tiên thử nghiệm in resin

Nhánh CPU tái sử dụng `head.workspace.html` đã dựng; không chạy lại RealDenseFace/Im2Haircut, không huấn luyện model mới. Giữ hình dạng đầu của từng người, không tạo đầu hình trụ hoặc sao chép tài sản LEGO.

## Một lần chạy trên Windows

Giải nén source vào project hiện tại, giữ nguyên `input`, `runs`, `.venv` và model cache. PowerShell tại thư mục project:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-figurine-cpu.txt
.\.venv\Scripts\python.exe scripts\run_figurine_v3.py --profile configs\figurine04.json
```

Đây là dependencies CPU tùy chọn; không cài vào `im2haircut`, `ml-depth-pro` hoặc `.venv-rdf`. Inference đọc model parser đã có, không tự tải weights. Profile trỏ workspace case04; nếu đã chuyển thư mục:

```powershell
.\.venv\Scripts\python.exe scripts\run_figurine_v3.py --profile configs\figurine04.json --workspace "D:\duong-dan\head.workspace.html" --parser-model "D:\AI-cache\face-parsing\resnet18.onnx"
```

Parser cần SHA256 `0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f`. Profile khóa checksum pixel ảnh trước. Workspace khác cần profile riêng; không ghi đè run cũ. Mở `runs/figurine_v3_.../review.html` offline; mở GLB bằng Blender/3D Viewer. STL dùng **mm, +Z hướng lên**; GLB dùng **m, +Y hướng lên**.

## Logic

| Phần | Thiết kế | Giới hạn |
|---|---|---|
| Đầu/mặt | Giữ vertex nguồn tại bản source_expression sau đổi tọa độ/kích thước | Kế thừa sai số reconstruction; chưa chứng minh đúng người |
| Tóc | Ba mask trước/trái/phải giới hạn vùng nhìn thấy; nền đặc bám đầu và các gờ cong bản lớn | Vùng khuất dùng prior; chưa fit ảnh sau; chưa suy luận ngôi tóc như người |
| Mắt | Giữ mí/hốc mắt nguồn; đổi cầu mắt thành trắng ngà, mống mắt, đồng tử và điểm sáng | Hướng nhìn là preset, chưa đo từ ảnh |
| Miệng gốc | Giữ biểu cảm nguồn, không thêm răng | Không sửa lỗi biểu cảm từ reconstruction |
| Cười custom | Mở môi cục bộ/nâng khóe, một dải relief nông liền theo bề mặt môi, vạch màu nhẹ | Biểu cảm giả lập; texture môi cũ có thể cần sơn lại |
| In | Đóng biên cổ/miệng trong, hợp nhất bề mặt Manifold để giữ mặt mượt, lấp khoang kín, kiểm tra một khối kín | Chưa kiểm tra slicer, support, độ dày, độ bền hoặc in thật |
| Màu | GLB và paint_guide.json | Saturn không in texture màu; cần sơn/hoàn thiện sau in |

Chiều cao **đầu kèm cổ 35 mm** là kích thước thử, chưa chốt sản phẩm; tổng cao có tóc sẽ lớn hơn. Prototype là khối đặc, chưa hollow. Độ phân giải LCD không phải cam kết chi tiết nhỏ nhất in được.

## Các tham số custom

Chỉnh `configs/figurine04.json` và chạy vào run mới:

- `head_neck_height_mm`: chiều cao đầu kèm cổ; cần điều chỉnh các kích thước chi tiết theo thành phẩm.
- `voxel_mm`: mặc định 0.23 mm để kiểm tra khối; nhỏ hơn tốn RAM/thời gian, chưa phải bảo đảm chất lượng.
- `hair.base_mm`, `ridge_mm`, `clumps`, `part`, `sweep`, `rgb`: nền, gờ, số gờ, ngôi, độ quét và màu. Ràng buộc ảnh vẫn giới hạn hình khối.
- `eyes`: mống mắt/đồng tử, màu, điểm sáng và gaze; mí mắt nguồn được giữ.
- `smile.enabled`, `gap_mm`, `corner_lift_mm`: xuất thêm cười tùy chỉnh; không tự đổi ảnh khép môi thành cười.

## Duyệt một lần

1. So source_baseline với source_expression: dáng mặt, tóc mai/chân tóc và mắt.
2. So trước, nghiêng, sau; phần sau tóc còn dùng prior.
3. Xem *.print_shape.glb và STL: đây là khối sau hợp nhất. Vạch răng và màu mắt không phải rãnh khắc.
4. Đọc report.json: topology, kích thước, biên được đóng và phần bị loại.
5. Chọn một biến thể để kiểm tra slicer và in thử. Watertight chưa đủ công nhận in thành công.

Model dựng đầu hiện tại được giữ làm baseline; parser chỉ tạo vùng ngữ nghĩa; tóc V3 là thiết kế procedural từ ảnh, không sinh sợi. Chưa có bằng chứng đổi sang model tạo 3D tổng quát sẽ giữ nét người/mesh in tốt hơn. Nếu cần đổi model đầu, kiểm chứng riêng trên ảnh/landmark.

Đây là kiểm tra khả thi hình khối và custom, chưa là hệ thống sản xuất đã xác nhận.
