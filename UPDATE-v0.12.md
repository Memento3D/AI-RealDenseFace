# Memento3D Head · Source v0.12.0

Bản source đầy đủ, gồm luồng MICA cũ, các experiment RealDenseFace đã gửi và workspace tối ưu texture mới. Chưa phải pipeline đầu hoàn chỉnh cho sản xuất.

## Cập nhật vào dự án Windows

Giải nén **nội dung** gói này vào `Memento3D-Head-Reconstruction`, cho phép cập nhật source trùng tên. Giữ nguyên `.venv`, `.venv-rdf`, `.venv-mica`, `vendor`, `input`, `runs`, cache model và cấu hình cục bộ của bạn. Gói không chứa các thư mục đó hoặc model weights. Không cần tải lại model hay cài lại CUDA.

Lệnh mới có thể chạy trực tiếp từ script, không cần cài lại package:

```powershell
.\.venv\Scripts\python.exe .\scripts\optimize_head.py `
    --head-run runs/rdf_open_20261023_152718_f3ce74/three_photo_7cf7011b
```

Bạn cũng có thể dùng `.venv-rdf` với cùng script. Phần mới chỉ dùng CPU, NumPy và Pillow. Nếu package đã cài editable trong chính thư mục này, lệnh tương đương là `python -m memento3d_head optimize-head ...`.

## Mở đúng output

Terminal trả về đường dẫn chính xác tới `head.workspace.html`, `head.masks.html` và `report.json`. Mỗi lần chạy tạo `head_workspace_<id>` mới. Header của viewer ghi nguồn `three_textures_<id>`; không mở lại file cũ từ lịch sử browser.

- Bên trái: texture cũ theo từng tam giác.
- Bên phải: mask được nhập, cân màu giới hạn, chuyển tiếp theo góc nhìn và màu nền.
- **Màu nền cho vùng thiếu**: bật/tắt để phân biệt phần có ảnh với phần dùng vật liệu mẫu.
- **Cân màu giới hạn**: bật/tắt để kiểm tra cân màu có làm sai sắc da không.
- **Bản đồ nguồn**: xanh dương là front, cam là left, tím là right; vùng màu trộn là nguồn chuyển tiếp. Đây không phải độ tin cậy được hiệu chuẩn.
- Link **Chỉnh mask và chọn màu da** mở công cụ xử lý cả ba ảnh.

## Một lượt review mask cho cả bộ ảnh

1. Ở `front`, tô **Loại vùng** trên tóc ảnh, tai nghe, áo và nền. Giữ da mặt, lông mày, mắt và môi.
2. Lặp lại ở `left` và `right`. Có công cụ **Khôi phục vùng**, chỉnh cỡ cọ và hoàn tác 8 nét gần nhất của mỗi ảnh.
3. Chuyển sang **Chọn màu da từ ảnh**, click một điểm má ít bóng. Màu nền phẳng chỉ làm kín phần chưa có dữ liệu, không phục hồi da thật phía sau đầu.
4. Bấm **Tải head_masks.json**. Browser tải file; lưu vào `configs/head_masks.json`.
5. Chạy:

```powershell
.\.venv\Scripts\python.exe .\scripts\optimize_head.py `
    --head-run runs/rdf_open_20261023_152718_f3ce74/three_photo_7cf7011b `
    --masks-json configs/head_masks.json
```

Chọn đúng JSON thuộc cùng nguồn texture. Mask có ràng buộc SHA-256; đổi nguồn `three_textures` cần review lại mask. Nếu có nhiều nguồn, cố định bằng `--texture-run runs/.../three_textures_<id>` cho cả hai lần chạy. Mask không tự ghi vào đĩa khi bạn tô; phải tải JSON.

Có thể chỉ định màu nền đã review qua `--skin-color '#B98D78'` (ví dụ cú pháp, không phải màu da được khuyến nghị cho bạn).

## Những thay đổi thực sự có trong bản này

- Một lệnh `optimize-head`, đọc lại export ba texture hiện có, không inference hoặc fit hình học lại.
- Bộ editor mask ba ảnh, chọn màu da thủ công và ràng buộc nguồn.
- Kiểm tra lại góc nhìn, depth và mask trước khi dùng màu.
- Cân gain RGB trong không gian tuyến tính, giới hạn 0.85–1.18, chỉ khi đủ điểm trùng; chưa tách ánh sáng khỏi màu da.
- Chuyển tiếp trọng số theo góc nhìn, ưu tiên front ở vùng nhìn thẳng. Có thể còn ghosting nếu các ảnh chưa đăng ký khớp; hãy kiểm tra mắt–môi.
- Vật liệu nền cho vùng thiếu và công tắc kiểm tra. Không sinh chi tiết giả cho vùng khuất.
- Viewer offline xem trước–sau, bản đồ nguồn và danh sách các vấn đề phải review.
- Report provenance, gain màu, số tam giác có ảnh, trạng thái mask, màu nền và các giới hạn.

## Phần còn cần hoàn thiện

| Hạng mục | Trạng thái v0.12 | Cách kiểm tra |
|---|---|---|
| Khuôn mặt có màu ba góc | Đã có bản xem | So ảnh gốc, kiểm tra hai góc 45 độ |
| Tóc ảnh/tai nghe/áo | Có mask thủ công, không tự tách | Review đủ ba ảnh và nhập JSON |
| Đường nối màu | Có gain giới hạn và chuyển tiếp | Bật/tắt cân màu, soi vùng má/thái dương |
| Vùng thiếu | Có màu nền phẳng | Tắt màu nền để xem phần thiếu thật |
| Mắt, môi, khoang miệng | Chưa sửa hình học | Xem chính diện và 45 độ |
| Tóc 3D và mũ tốt nghiệp | Chưa ghép | Chuẩn bị asset mẫu sau khi chốt mặt |
| Atlas UV và GLB có texture đã bake | Chưa thực hiện trong workspace mới | OBJ cũ chỉ có projective/planar UV |
| Likeness, vật liệu PBR, đầu kín để in | Chưa xác minh | Cần review và quy trình riêng |

Gửi một lượt: `report.json` của workspace mới, screenshot chính diện/±45°, và mask JSON nếu có vùng cần sửa. Không dùng tỷ lệ phủ tam giác làm tỷ lệ giống khuôn mặt.

## Kiểm tra

131 unit tests đạt, architecture check đạt, JS hai trang mới qua syntax check; có test phép chiếu, mask loại vùng, gain màu giới hạn, xuất workspace và từ chối mask nguồn cũ. Chưa chạy với model/mesh thật của bạn trong môi trường này; CUDA không cần cho lệnh mới, hình ảnh WebGL cần review trên máy bạn.
