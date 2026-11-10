# Chỉnh asset tóc nam — lượt 01

Review: `runs/male-hair-refined-01/review.html`. Đối chứng là cage-lift ở male-hair-cage-01. Output và report mới, không ghi đè nguồn. Script Blender bổ sung `--refine-asset` cùng `--cage-fit`.

## Đã áp dụng

Trước khi bind, xoay hướng vertex tóc về phía dưới ở vùng sau theo tọa độ cực quanh seed center: tăng góc cực tối đa 0,30 rad, trọng số theo hướng về gáy, giảm ảnh hưởng gần đỉnh. Chỉ xử lý tóc. Đây chưa phải vùng root được gán nhãn hay fit giải phẫu tự động.

A (cage-close): khung ôm, Smooth factor 0,5 × 8 iterations. B (cage-lift): khung nâng, hair-only voxel remesh với voxel 0,0012 × scale m và Smooth factor 0,5 × 4 iterations. Hai bản khác cả độ nâng lẫn xử lý mesh, nên chưa thể quy toàn bộ khác biệt hình ảnh cho riêng remesh. Không sửa đầu hoặc texture.

live-cage.blend lưu binding trước remesh/smoothing; editable.blend và GLB chứa kết quả sau xử lý. Không coi live-cage là bản hoàn thiện của bước làm mượt.

## Kết quả thực tế

Audit nạp lại GLB: hai bản giữ geometry/UV đầu ở mức 1e-6 và pixel texture chính xác so với đầu nguồn đã chốt.

- Đối chứng: 129.750 tam giác tóc, 118 thành phần, 0 cạnh nonmanifold.
- A: 129.750 tam giác, 118 thành phần, 3 cạnh nonmanifold trên bản audit weld 1e-7; không chọn cho hướng in tại thời điểm này.
- B: 459.216 tam giác, 52 thành phần, không có cạnh hở/nonmanifold theo kiểm tra; watertight theo kiểm tra topology. Chưa phải một khối nối liền.

Tóc sau phủ thấp hơn, nhưng lọn vẫn dạng lá và đường nối với tóc nền còn rõ. Chưa có bằng chứng rằng tạo hình đã đạt mục tiêu người thật. Remesh giảm số thành phần nhưng tăng đáng kể số tam giác; không coi đây là tối ưu hoàn chỉnh.

## Chưa được chứng minh và bước tiếp

Chưa đo độ dày nhỏ nhất, tự giao, phần giao/nối với đầu, island/support trong slicer. Chưa in thực tế. Đầu cao 100 mm trong audit chỉ là phép quy đổi minh họa.

Ưu tiên xác định vùng gốc từng cụm, tạo lớp nối bên trong có kiểm soát để gom thành phần mà không phủ đè mặt/tai, rồi kiểm tra độ dày theo kích thước đã chốt. Không tiếp tục smoothing toàn bộ với hy vọng thay đổi phong cách gốc của asset. Nếu cần tóc người thật hơn, phải sửa hình khối/hướng chải của một mẫu chủ hoặc dùng asset có dáng phù hợp hơn.

Architecture PASS. Unittest log: `runs/male-hair-library-01/unittest-refined.log`; suite còn lỗi, không tuyên bố toàn bộ kiểm tra đạt.
