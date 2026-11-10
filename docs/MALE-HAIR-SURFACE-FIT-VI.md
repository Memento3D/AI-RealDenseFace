# Fit tóc theo mặt sọ — thử nghiệm v2

## Phạm vi

Thay affine fit cố định bằng radial shell transport: cùng asset và cùng policy được chạy trên input07, input04, capture02. Đây là thử nghiệm cho các đầu cùng hệ trục, đơn vị và cấu trúc nguồn hiện tại; chưa chứng minh dùng được cho mọi đầu hoặc mọi kiểu tóc.

Đầu nguồn vẫn là `runs/buzz-framework-unified-01/attempt-02/<case>/gap-0_002/head.glb`. Không chỉnh khuôn mặt, UV hoặc ảnh texture để làm vừa tóc. Asset tóc gốc không đổi. Output riêng tại `runs/male-hair-surface-02/`.

## Công thức

Với hướng đơn vị d từ tâm sọ c tới vertex tóc seed, lấy bán kính mặt đầu R(d) bằng BVH ray cast. Tóc seed được chuẩn hóa cùng hệ trục trước. Ellipsoid tham chiếu có bán trục a,b,c; bán kính theo hướng d là:

`E(d) = 1 / sqrt((dx/a)^2 + (dy/b)^2 + (dz/c)^2)`.

Vị trí mới:

`p' = center_head + d * [R(d) + gap + relief * (r_seed - E(d))]`.

Giữ offset **có dấu** giữa tóc và ellipsoid để không ép cả mặt trong lẫn mặt ngoài lên cùng một bề mặt. V1 đã clamp offset về 0 và gây nếp ép/rối ở bên đầu, không dùng làm bản đề xuất.

Scale s = chiều rộng vùng đầu trên / 0,18. Ellipsoid = s × (0,080; 0,090; 0,080) m. Gap = 0,004s m. Relief thử 0,75 và 1,00. Tâm sọ và scale được ước lượng từ đỉnh và bounds vùng đầu trên, chưa phải mốc giải phẫu được hiệu chuẩn. Các hệ số là heuristic thử nghiệm, không phải công thức từ nghiên cứu đã được xác nhận. Chi tiết từng đầu và số ray miss nằm trong report.json.

Phương pháp không tự tạo lọn ở vùng asset thiếu tóc. Chưa xử lý tự giao, tiếp xúc tai hay biến dạng thể tích cục bộ. Chưa có vùng gốc tóc được tác giả gán nhãn, nên chưa xem đây là cage fit hoàn chỉnh.

## Khuôn mặt và cách hiển thị

Lượt male-hair-fit-03 render đầu bằng ánh sáng Cycles, khác trang cũ hiển thị texture unlit. Ánh sáng làm cảm giác khuôn mặt và màu da thay đổi. V2 render phần đầu unlit, tóc có ánh sáng; không dùng render này làm bằng chứng độ giống.

File GLB xuất giữ material đầu gốc; material unlit dùng riêng cho ảnh review. Audit nạp lại GLB, so các góc tam giác gồm tọa độ thế giới và UV ở độ chính xác 1e-6; so pixel RGBA texture chính xác. Xem roundtrip-audit.json để biết những trường hợp thực sự vượt kiểm tra. Đây không phải phép đo identity/likeness hoặc so với mọi phiên bản đầu trước đó.

## Xem và chạy lại

`runs/male-hair-surface-02/review.html` tổng hợp ba đầu. Các thư mục surface-low/surface-volume có GLB và editable.blend. Script `tools/headapi/male_hair_fit_trial.py` nhận `--surface-fit --case input07 --output <thư mục mới>` khi chạy trong Blender nền, dùng `--disable-autoexec`. Không ghi đè attempt cũ.

Tiếp theo cần chọn dáng phù hợp rồi chỉnh asset một lần trong Blender: gom lọn nhỏ, làm đầu lọn bớt nhọn, xác định vùng gốc, kiểm tra khoảng hở và độ dày theo kích thước in. Chưa nối API, chưa kết luận in được.

## Kết quả chạy

Đã chạy input07/input04/capture02, mỗi đầu hai mức relief. Sáu trường hợp đều 0 ray miss. Audit roundtrip cả sáu: geometry/UV corners bằng ở mức làm tròn 1e-6; pixel RGBA texture bằng chính xác. Nguồn đối chiếu là đầu đã chốt buzz-framework-unified-01, không phải mọi output lịch sử.

Quan sát: v2 giảm nếp ép ở hai bên so với v1; các lọn nằm trên bề mặt sọ nhiều hơn nhưng tạo hình vẫn dạng lá, đỉnh còn nhọn. Vùng tóc nền lộ rõ là đặc điểm thiết kế chưa hoàn thiện. Chưa duyệt hình dáng hoặc cơ chế fit cho dùng chính thức.

Lưu ý provenance: report của lượt chạy v2 có trường surface_policy=radial_shell_transport_v2 đúng với thuật toán, nhưng nhãn status còn kế thừa chữ AFFINE từ script trước lúc sửa nhãn. Đây là lỗi nhãn report, không phải kết quả affine; giữ nguyên report cũ và ghi rõ tại đây. Script hiện đã sửa nhãn cho các lượt tương lai.

Đã kiểm tra thêm GLB male-hair-fit-03/compact so với đầu 07 đã chốt: geometry/UV ở mức 1e-6 và texture pixel đều giữ nguyên. Bằng chứng: `runs/male-hair-surface-02/previous-head-audit.json`. Như vậy chưa thấy biến đổi mesh/texture ở nhánh này; khác biệt cảm nhận từ render cần đánh giá cùng camera/ánh sáng.

Kiểm tra phần mềm: architecture PASS. Log unittest cuối: `runs/male-hair-library-01/unittest-surface-final.log`; suite vẫn có lỗi tồn tại, không coi là toàn bộ kiểm tra đạt.
