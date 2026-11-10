# Chuyển sang thư viện tóc dựng sẵn

Theo định hướng người dùng ngày2026-11-04: tạm dừng nhánh tự sinh shell/cụm tóc. Giữ đầu nền cá nhân và các kết quả thử để đối chiếu, không tiếp tục tinh chỉnh chúng như phương án chính. Mục tiêu mới: tuyển asset tóc có tạo hình tốt, chuẩn bị preset Blender có thể fit nhiều đầu và kiểm tra riêng cho in.

## Ứng viên ưu tiên

[BaseMesh Pack Free Edition — blenderustad](https://superhivemarket.com/products/basemesh-pack-free-edition-character-related-models): trang publisher ghi5 hair models, file.blend và giá0. Mục License dẫn tới trang [Creative Commons](https://superhivemarket.com/page/creative-commons-license) mô tả CC-BY4.0. Cần kiểm tra metadata/license đi kèm file khi tải; chưa tải/chưa kiểm tra mesh, chưa khẳng định5 mẫu phù hợp hoặc in được. Ưu tiên kiểm tra một mẫu ngắn thay vì cài cả thư viện lớn.

[FREE Male Fashion Hair Collection01 — Vincent Freelance](https://sketchfab.com/3d-models/free-male-fashion-hair-collection-01-lowpoly-2735370f6f764078a18f5d80651050a2): tìm được trang collection, đọc trực tiếp bị403; chưa xác minh giấy phép/download/geometry. Chỉ là ứng viên dự phòng, không tự coi tênFREE là quyền tái sử dụng.

Không quay lại dùng mặc định ba mẫu MakeHuman Hair01 đã thử: có mesh hở/nhiều mảng, và một mẫu có xung đột license giữa trang và metadata. Các kiểm tra cũ ở runs/hair-cc0-selection-01.

## Preset Blender cần có

Mỗi mẫu: nguồn, giấy phép, tác giả, checksum, reference head/scalp, mesh nguyên bản và bản chuẩn bị in riêng. Chuẩn hóa đơn vị/trục/origin. Mốc căn: đỉnh sọ, trán, thái dương, trước/sau tai và gáy. Không sửa geometry khuôn mặt để làm tóc vừa.

Fit hai lớp: căn transform chung, rồi lattice/cage điều chỉnh vùng chân tóc theo scalp. Giữ thể tích/dáng tóc bên ngoài; không shrinkwrap toàn bộ bề mặt ngoài. Tham số dự kiến: rộng/sâu/cao đầu, vị trí chân tóc, chiều dài mái, độ phồng đỉnh, ôm hai bên, dài gáy. Một tham số chỉ công bố sau khi asset thực tế có thể điều khiển ổn định.

HeadBase và HairStyle là hai component riêng. Mai/gáy nền hiện có được giữ, tránh đè tóc mới lên tai. Tạo bản in khác bản chỉnh sửa; kiểm tra phần rời, self intersections, độ dày tại kích thước tượng và slicer. Mesh đẹp hoặc có STL không tự chứng minh in được.

## Điều kiện đạt

Đầu tiên chọn một asset đẹp ở vật liệu xám trên chính đầu tham chiếu của nó. Sau đó fit lên07/04/capture02 cùng preset, kiểm tra camera nhiều góc và bảo toàn đầu. Nếu phải sửa nhiều bằng tay mỗi đầu, chưa gọi là fitter tự động. Không hứa fit mọi đầu; cần nêu phạm vi đầu/phong cách đã test và cơ chế từ chối case không phù hợp. Chưa tải trọng sốAI hoặc chuyển GPUcloud cho nhiệm vụ này.
