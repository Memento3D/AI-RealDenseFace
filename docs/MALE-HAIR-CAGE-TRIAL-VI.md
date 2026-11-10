# Đầu 07 — thử khung Surface Deform

Review: `runs/male-hair-cage-01/review.html`. Giữ đối chứng radial v2 và thử cage-close/cage-lift. Không ghi đè các lượt trước, không thay API. Code adapter Blender: `tools/headapi/male_hair_fit_trial.py --cage-fit --case input07 --output <folder mới>`.

## Cơ chế đã chạy

Tạo proxy ellipsoid 610 vertex, chuẩn hóa theo chiều rộng vùng sọ trên; bind tóc với proxy bằng Surface Deform trước khi biến dạng. Ray từ tâm ước lượng tìm điểm mặt đầu; dịch proxy theo pháp tuyến với độ nâng theo vùng:

`crown=max(dz,0)^2; fringe=max(-dy,0)^2*max(dz,0)`.

Close: `h=s*(0,0015+0,002*crown)` m.

Lift: `h=s*(0,002+0,009*crown+0,004*fringe)` m.

`C'=hit_head + h*normal`. Tóc đi theo binding của proxy, không ép trực tiếp từng vertex lên mặt đầu. Tham số là heuristic, chưa calibrated. Có 42/610 ray miss ở mỗi bản, giữ vị trí seed tại điểm miss. Chưa có hệ mốc giải phẫu trán–tai–gáy hay giới hạn ảnh hưởng vùng gốc được gán nhãn; không coi đây là thuật toán phổ quát đã hoàn thiện.

Mỗi bản có live-cage.blend giữ binding để điều chỉnh; editable.blend và GLB là mesh đã evaluate. Proxy không xuất vào GLB. Material đầu gốc được giữ khi xuất; material emission chỉ dùng để render đầu unlit nhằm so với ảnh review cũ.

## Kiểm chứng

`runs/male-hair-cage-01/audit.json`: hai GLB sau xuất có tọa độ thế giới/UV góc tam giác khớp đầu nguồn khi làm tròn 1e-6, pixel RGBA texture bằng chính xác. Không thay khuôn mặt để làm vừa tóc. Hash file đầu và blend nguồn giữ nguyên.

Mỗi mesh tóc có 129.750 tam giác, 118 thành phần sau weld tọa độ 1e-7 trên bản audit; không có boundary hoặc edge hơn hai mặt, winding consistent và không phát hiện tam giác area dưới 1e-14. Các phép này không phát hiện đầy đủ self-intersection hoặc chứng minh thành phần rời không giao nhau.

Ví dụ quy đổi lấy chiều cao đầu 100 mm được ghi trong audit, chỉ là kịch bản minh họa, không phải kích thước sản phẩm đã chốt. Chưa đo bề dày nhỏ nhất hoặc kiểm tra print union, island/support trong slicer. Không có chứng nhận sẵn sàng in.

## Đánh giá và hướng tiếp

Bind hoạt động, có thể điều chỉnh độ ôm/độ nâng thông qua khung mà không sửa đầu. Tuy nhiên cả hai bản vẫn giống tóc phủ đỉnh, lọn nhọn và vùng sau kết thúc đột ngột; chưa đủ bằng chứng rằng cage đẹp hơn radial. Không thêm chi tiết sợi ở giai đoạn này.

Cần chuẩn bị asset một lần: xác định vùng gốc của các lọn, giảm ngọn nhọn, gom khối, tạo chuyển tiếp theo vùng sau/hai bên; rồi mới gán các mốc trán–tai–gáy để tái sử dụng. Bản dành cho in cần nối có kiểm soát giữa lọn và đầu hoặc shell gốc, kiểm tra topology và độ dày theo kích thước thực. Giữ component tách riêng ở bản biên tập.

Architecture check PASS; unittest log: `runs/male-hair-library-01/unittest-cage.log`. Suite còn lỗi; không dùng bind thành công để tuyên bố toàn bộ pipeline đạt.
