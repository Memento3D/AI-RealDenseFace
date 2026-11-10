# Thư viện tóc nam — asset đã nhận

Asset gốc nằm tại `assets/hair-library/Ultimate pack Free version.blend` và `assets/hair-library/texture.zip`, ngoài runs và được ignore Git. Không sửa file gốc.

SHA-256 blend: `2ca8b702b04341bec9a5485e5ae2384c1187523de391f7a8de01e95c316676cc`.

Đã kiểm tra 20 object; có Male_Hair và Female_Hair. Mẫu nam có 65.111 vertex, 65.181 polygon, không có boundary/nonmanifold edge trong base mesh theo inspector; không có modifier hoặc alpha link được phát hiện. Chưa kiểm tra self-intersection và độ dày thực tế, chưa kết luận in được.

Report: `runs/male-hair-library-01/inspection-02/inspection.json`.
Bản tách: `runs/male-hair-library-01/male-source-01/male-hair.glb` và `male-hair-source.blend`. Texture lấy chọn lọc từ ZIP. Checksum nguồn giữ nguyên.
Xem ba góc tại `runs/male-hair-library-01/review.html`.

Mẫu hiện có nhiều lọn nhọn, thiên về tạo hình nhân vật. Màu render là material nguồn, chưa phải màu cuối. Chưa fit vào đầu cá nhân. Hair object nằm quanh origin trong khi body tham chiếu đặt đứng đầy đủ; không dùng vị trí hai object làm căn cứ ghép trực tiếp.

Bước tiếp: chuẩn hóa trục và mốc chân tóc; tạo bản sao tóc nam ngắn, giảm ngọn nhọn và gom lọn; dựng cage vùng gốc theo trán–thái dương–tai–gáy, giữ độ phồng lọn. Thử đầu 07 trước rồi 04/capture02; xác minh geometry/texture đầu giữ nguyên. Sau khi duyệt dáng mới kiểm tra thành phần rời, self-intersection và bề dày theo kích thước mm dự kiến. Chưa nối API.

Đầu nền giữ tại `runs/buzz-framework-unified-01/attempt-02/input07/gap-0_002/head.glb`. Tóc ngoài là component riêng; giữ lớp tóc nền đỉnh–mai–gáy.

Nguồn: BaseMesh Pack Free Edition của blenderustad, https://superhivemarket.com/products/basemesh-pack-free-edition-character-related-models . Chưa nhận file license riêng; giữ điều khoản và attribution trước khi phân phối asset.

Architecture check PASS. Unittest: 562 tests, 2 failures, 4 errors, 1 skipped; log `runs/male-hair-library-01/unittest-asset.log`. Chưa phải suite xanh.

## Thử fit đầu 07

Review mới: `runs/male-hair-fit-03/review.html`, gồm đầu nền, khối lớn và compact, mỗi bản bốn góc. GLB và editable.blend nằm trong từng thư mục biến thể. Script: `tools/headapi/male_hair_fit_trial.py`.

Đã chuẩn hóa transform của object trước khi fit. Đây là affine fit thủ công bằng kích thước/offset có ghi trong report, chưa phải preset tự thích nghi mọi đầu. Lượt 01 bị transform nguồn làm lệch vị trí, không dùng để đánh giá. Lượt 02 chuẩn hóa transform; lượt 03 hạ bản compact gần trán hơn. Không ghi đè các lượt.

Đánh giá: compact hợp lý hơn ở chính diện nhưng ở góc bên nhiều phần tóc chìm vào sọ; chưa phủ hài hòa phía sau. Lọn còn dạng lá, nhọn, thiên về nhân vật. Không chốt đây là kiểu tóc hoàn thiện. Bước tiếp cần cage fit vùng chân tóc theo mặt sọ, giữ độ phồng lớp ngoài, chỉnh riêng vùng sau và hướng chải; không scale toàn khối thêm vì dễ làm mái hoặc hai bên quá lớn. Có thể custom asset này, nhưng chất lượng hình dáng gốc vẫn giới hạn độ hiện thực.

Kiểm chứng trong report: geometry và UV đầu không đổi trong scene; checksum file đầu và blend nguồn giữ nguyên. Chưa đo clearance/self-intersection/độ dày hoặc kiểm chứng in. Không tuyên bố giữ độ giống khuôn mặt chỉ từ việc không sửa mesh.

Xem thử nghiệm fit tái sử dụng mới: `docs/MALE-HAIR-SURFACE-FIT-VI.md` và `runs/male-hair-surface-02/review.html`. Đầu được render unlit để đối chiếu texture; đã audit GLB sau xuất trên ba đầu.
