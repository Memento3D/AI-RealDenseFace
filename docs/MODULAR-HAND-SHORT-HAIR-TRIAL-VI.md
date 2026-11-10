# Thử bàn tay đầy đủ, tóc ngắn và mũ tốt nghiệp

## Kết quả hiện tại

Trang xem: `runs/hand-hair-assembly-01/review.html`. Bản xuất được kiểm tra là `attempt-03`; giữ nguyên các lần thử trước. Chưa nối API.

- Loại phần tay cũ bị cắt; thay bằng asset bàn tay đầy đủ năm ngón, dựng rig 16 xương để chỉnh tư thế. GLB chứa tay đã áp dụng tư thế; Blender giữ rig có thể chỉnh.
- Thêm một preset tóc ngắn gồm nền da đầu và 23 lọn dạng mesh. Đây là kiểu tóc thiết kế sẵn, không suy luận tóc cá nhân từ ảnh.
- Chỉnh kích thước/vị trí mũ, thêm dây tua mới, thêm viền cổ tay áo để che chỗ ghép.
- Kiểm tra độc lập cho thấy ảnh texture đầu giữ nguyên từng pixel; sai lệch tọa độ tam giác đầu tối đa khoảng 3,21e-7 trong đơn vị làm việc. Sai lệch GLB tay so với mesh Blender sau đánh giá rig khoảng 6,87e-8.

## Nguồn và khả năng tái hiện

Bàn tay `Hand  - Realistic` của Dan Ulrich trong [Blender Human Base Meshes](https://www.blender.org/download/demo-files/), giấy phép CC0. Asset nền có 831 đỉnh; mỗi ngón có chuỗi ba xương, cộng một xương lòng bàn tay. Không cài thêm mô hình AI hoặc tải trọng số trong bước lắp ghép này.

Builder: `tools/headapi/modular_hand_hair_trial.py`, chạy bằng Blender nền với `--output` trỏ tới thư mục lần thử mới. Dữ liệu chuẩn bị, cấu hình và checksum nằm trong `runs/hand-hair-assembly-01`; không đưa ảnh, GLB, môi trường hay asset vào Git.

`attempt-01` có viền ghép cổ tay chưa ổn và tóc dày. `attempt-02` dừng vì giả định cổ tay nguồn là biên hở không đúng: asset có nắp đóng. `attempt-03` xử lý vùng cuối cẳng tay có nắp và thêm viền áo, đồng thời giảm bề dày tóc mái.

`audit.json` kiểm tra checksum nguồn, hình học/texture đầu và mesh tay GLB so với Blender. `attempt-03/rig-audit.json` kiểm tra trọng số, năm chuỗi ngón và sai số đích IK sau khi mở lại file. Đích IK là đầu xương, chưa phải xác nhận bề mặt đầu ngón tiếp xúc đúng vành mũ.

## Đánh giá hình ảnh và việc tiếp theo

Bàn tay không còn phần ngón bị cắt, nhưng dáng còn phẳng và cần chỉnh độ cong đốt/ngón, chiều dày, màu da để đồng nhất với đầu. Tóc mái vẫn lặp đều, giống các tép: cần chuyển thành ít khối lớn bất đối xứng, hướng vuốt rõ và đường chân tóc mềm hơn. Chưa nên coi bản này là tạo hình hoàn thiện.

Cổ tay được che bằng viền áo, chưa boolean union với thân. Chưa kiểm tra đầy đủ giao cắt giữa tay–mũ, tóc–mũ, tự giao cắt, độ dày hoặc khả năng in. `ready_for_machine=false`. Giữ từng component riêng để sửa; chỉ tạo mesh in sau khi dáng và tiếp giáp được chấp nhận.

Hướng tiếp: chỉnh pose bàn tay trên rig, đổi tóc mái thành vài khối vuốt ngắn, render cùng góc để đánh giá; sau đó mới kiểm tra tiếp xúc/giao cắt, chuẩn bị mesh in và cuối cùng tích hợp API.

## Kiểm tra repository

Architecture boundaries: PASS. Unittest discover: 562 tests, 1 failure, 4 errors, 1 skipped. Các lỗi gồm bài resume giả lập, manifest hair và môi trường im2; chi tiết lưu ở `runs/hand-hair-assembly-01/unittest.log`. Không tuyên bố toàn bộ suite đã đạt.
