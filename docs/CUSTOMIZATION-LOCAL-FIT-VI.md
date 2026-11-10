# Thử nghiệm local: kính + mũ hoặc tóc (2026-11-05)

Đã xuất 4 bản phối trên model hiện tại và 4 bản kiểm tra hình học từ hai bundle RDF có sẵn. Chưa tích hợp API, BE, B2 hoặc màn hình custom của sản phẩm.

## Mở bộ so sánh

Trong PowerShell:

```powershell
cd D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction
.\scripts/review/START-CUSTOMIZATION-REVIEW.ps1
```

Mở `http://127.0.0.1:8092/review-site/`. Có nút chính diện, góc 3/4, bên hông, phía sau, toàn thân và ẩn thân/tay. Chỉ tải một model mỗi lần; khi đổi mẫu, viewer giải phóng geometry/material/texture cũ. Ctrl+C dừng server. Cổng này dành cho xem local; không cần ngrok và không thay đổi AI API.

Ảnh so sánh: `http://127.0.0.1:8092/review-site/comparison.html`.

- Bộ model có mặt/texture hiện tại: `runs/customization-prep-20261105/fit-review-06/`.
- Bộ kiểm tra mesh xám: `runs/customization-prep-20261105/cross-head-03/`.
- Kiểm chứng nguồn/mặt: `runs/customization-prep-20261105/review-site/audit.json`.
- Thư viện 29 kính và 3 tóc: `assets/custom-library/prepared-20261105/review.html`.

## Kết quả hiện tại

| Bản phối | Số điểm có khoảng hở có dấu < -1 mm | Dung lượng phụ kiện riêng |
| --- | ---: | ---: |
| Mũ gốc + kính tròn 03 | 0 / 1008 | 46.328 byte |
| Tóc side-part + kính tròn 03 | 0 / 1008 | 53.756 byte |
| Tóc upcomb + kính tròn 03 | 0 / 1008 | 56.180 byte |
| Mũ gốc + kính mẫu 01 | 0 / 1200 | 53.960 byte |

Kính tròn ban đầu có 132 điểm dưới -1 mm; sau điều chỉnh độ rộng và chỗ đỡ tai còn 0. Giá trị nhỏ nhất vẫn khoảng -0,694 mm, nên đây không phải kết luận không còn giao nhau. Hai bundle kiểm tra bổ sung còn 4 điểm dưới -1 mm mỗi bản. Hai bundle có hash hình học khác nhau nhưng hình dạng gần nhau; không coi đây là kiểm chứng trên hai người độc lập hoặc nhiều kiểu khuôn mặt.

Phép đo tìm điểm gần nhất trong 32 tam giác có tâm gần nhất rồi lấy dấu theo pháp tuyến cục bộ. Tai/mesh hở làm dấu khoảng cách có thể gây cảnh báo giả. Chưa kiểm tra giao nhau toàn bộ các tam giác, mesh kín, độ dày để in hoặc độ bền của gọng.

## Cách căn để có thể tái sử dụng

1. Đọc 68 landmark FLAME bằng barycentric binding trong `projection_bundle.npz`; không dùng tọa độ pixel hoặc đo riêng trên ảnh hiện tại.
2. Tạo hệ trục mặt từ hai mắt và cằm. Đặt kính theo vị trí mắt và sống mũi, giữ nguyên hình dạng mặt.
3. Lấy mốc đỡ gọng từ phần trên của semantic mask tai trái/phải. Cân bằng khoảng cách mắt với độ rộng hai tai; nếu phải phóng kính thêm quá 20% thì đánh dấu cần mẫu kính rộng hơn.
4. Cho phép dịch nguyên kính ra trước tối đa 6 mm, nâng/hạ đoạn gọng tối đa 12 mm và nới gọng sang bên tối đa 8 mm. Gọng được chỉnh cục bộ; phần trước của kính giữ nguyên hình dạng sau phép scale/pose.
5. Mỗi asset cần profile đã rà soát về trục, tâm tròng và độ cao cầu kính. Chưa tự suy ra profile cho toàn bộ 29 kính hoặc asset bất kỳ.
6. Mũ hoặc tóc là hai lựa chọn loại trừ nhau. Kính là lớp độc lập nên không cần chạy lại AI khi đổi kính. Tóc hiện chỉ khởi tạo bằng bao da đầu, chưa có bộ giải va chạm hoàn chỉnh.

Code: `tools/customization/fit_math.py`, `fit_trial.py`, `cross_head_trial.py`, `fit-profiles.json`. Mỗi lần thử phải dùng thư mục output mới. Script kiểm tra hash asset, nguồn job và giữ báo cáo transform/clearance; không cập nhật pin của API.

## Mũ gốc và bảo toàn mặt

Mũ/tua rua được lấy từ tam giác và UV của `runs/prod-win-01/src/GLB/men.glb`, thay cho mũ phẳng dựng thủ công. Giữ độ nghiêng, kích thước, texture và tua rua màu kem ở bên trái; không tự tô lại màu. Kiểm chứng tam giác phần mũ đã chọn sau export: sai số vị trí 0; pixel texture giống nguồn.

Đầu hiện tại được tịnh tiến lên 18 mm trong bản local để thử đội mũ, không đổi tỉ lệ/biến dạng mặt. UV và pixel texture giống đầu của job hiện tại; sai số vị trí sau khi trừ phép tịnh tiến nhỏ hơn 3e-8 m. Đoạn cổ được nối lại giữa biên thật của body gốc và biên đầu, không dùng lại connector của body cũ.

Giới hạn còn thấy khi render: mask phía sau mũ có vùng hở cần tách sạch hơn; bề mặt và màu nối cổ chưa đều. Tóc low-poly còn giống khối mũ/helmet, cần chỉnh hairline và nâng chất lượng asset. Texture đầu cũ vẫn có vùng tóc đã tô, cần có chính sách scalp/appearance riêng khi đổi kiểu tóc. Bản tóc dùng body/bàn tay đầy đủ của preset hiện có để tránh mất tam giác bàn tay chạm mũ; cần thống nhất body/pose trước khi tích hợp custom. Kính mẫu 01 giữ vật liệu gốc nên phản sáng mạnh; kính tròn 03 dùng gọng charcoal và bỏ tròng ở bản xem fit để tránh hiện đục. File gốc của tất cả asset vẫn được giữ riêng.

## Hướng tích hợp sau khi duyệt output

Lưu base model, head, các accessory và profile/transform riêng, thay vì xuất lại toàn bộ GLB mỗi lần đổi kính/tóc. Các sidecar `accessories.glb` hiện khoảng 45–55 KiB, giúp web/app tải và cache độc lập. Composite trong bộ review còn là bản local giữ nhiều geometry/texture gốc, không coi là ngân sách tải mobile cuối cùng. Giữ master riêng cho quy trình in; giảm dung lượng preview không tự động áp dụng vào master.

Trước khi nối API cần: xử lý mask mũ/cổ và hairline; thống nhất body/pose; rà gọng ở hông/phía sau tai; kiểm tra các ảnh/người đa dạng và tình huống landmark lỗi; sau đó mới công bố asset đã hỗ trợ và trả transform/asset ID. Không tự báo fit thành công nếu vượt giới hạn.

## Kiểm tra

- 9 test mới về pose/scale, mốc mũi/tai, giới hạn biến dạng, độ rộng đầu, dữ liệu lỗi và mũ/tóc loại trừ nhau: PASS.
- `scripts/check_architecture.py`: PASS.
- Full suite: 579 tests, 4 errors, 1 skipped. Các lỗi có sẵn liên quan manifest thiếu `OPEN-HAIR-ON-HEAD.ps1`, môi trường im2/WSL và thiếu `pyhocon`; xem `full-tests-02.log`. Không coi full suite là đã pass.
- 8 GLB và 32 ảnh Blender nhiều góc đã xuất; kiểm tra mắt thường các bản chính, kiểm chứng source cap và head round-trip. Viewer đã bundle Three.js có sẵn trong FE; không cần tải thư viện từ CDN.

Nguồn asset mới: iPoly3D lowpoly glasses pack và ba kiểu tóc của Micket từ OpenGameArt, CC0; bằng chứng nguồn/license/checksum trong `assets/custom-library/source-batch-20261105/`. Không có asset nào được chứng nhận sẵn sàng in.

Khi xem riêng cap trên mesh xám còn thấy các mảnh ngón tay của nguồn ở vùng chạm mũ: mask chưa tách thành accessory độc lập hoàn toàn. Đây là lỗi cần sửa trước khi dùng chung một body để bật/tắt mũ, không phải lỗi của ảnh mới hoặc của phép căn kính.
