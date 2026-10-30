# Head24.2 — tách bbox ảnh bên khỏi landmark bắt buộc

## Bằng chứng từ lần chạy 05 ngày 29/10/2026

Front: 14 observations, 0 bị lọc, 1 mặt khoảng 190.7x218.9 native pixels.
Left: 14 observations, 4 bị lọc mép/không hợp lệ, 1 mặt khoảng188.4x220.4 pixels.
Right: 0 observations trước bộ lọc sau74calls. Không phải ngưỡng64pixel loại mặt.
Đây là giới hạn phát hiện của MediaPipe trên ảnh này, chưa chứng minh lý do cụ
thể là yaw, kính hay ánh sáng. Không suy ra ảnh chắc chắn phải chụp lại.

## Cài/chạy

Giải nén toàn bộ gói này đè lên các file Head24 ở thư mục project hiện tại.
Giữ nguyên source V2.3, vendor và môi trường. Gói đã cập nhật checksum tương ứng.

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD24.ps1 -InputDir ".\input\05"
```

Không cài package mới hay tải weights mới. Các lần chạy cũ được giữ nguyên.

## Thay đổi

Chỉ khi không có bbox landmark ở left/right, gọi worker FaceBox trong .venv-rdf.
Dùng FaceDetector chính trong vendor/models/facebox/detector.py, weights đã có
và SHA256 đã pin. Worker chạy CPU float32, tắt torch.compile, quét cửa sổ giống
trước; score>=0.6. Pad vuông để tránh bóp tỉ lệ; đổi bbox về pixel nguồn, loại
bbox trong padding hoặc sát mép tile, gộp trùng và chọn chủ thể bằng policy cũ.
Không thay bằng bbox toàn ảnh khi detector thất bại.

Front vẫn bắt buộc landmark MediaPipe để kiểm tra tâm mắt độc lập với RDF.
Ảnh bên dùng FaceBox có `landmarks_crop: null`, `roi_backend: facebox`, score
và `face_box_crop` thật. Không tạo468 điểm giả, không coi bbox là landmark.
Crop và texture lấy từ ảnh gốc. debug right.landmarks.png sẽ có hộp ROI vàng
và dòng 'no profile landmarks', không có các điểm xanh như front/left.

Sau joint fitting, ảnh bên dự phòng phải qua kiểm tra bbox chiếu so với ROI
FaceBox: khoảng05..95% của các vertex contour được kiểm tra tỉ lệ và tâm.
Ngưỡng này là heuristic bắt lỗi lớn, không chứng nhận landmark profile hay độ
chính xác giải phẫu. Kiểm tra camera explicit, biến dạng đầu trên5px và các
kiểm tra hình học/tóc vẫn giữ nguyên. Không có bằng chứng in thật.

## Báo cáo khi dừng

- `01_crops/right.facebox.json`: score lớn nhất, số detections>=0.6, số bbox bị
  loại vì padding/mép, bbox candidates và source/weights SHA.
- `01_crops/right.facebox.log`: log worker CPU.
- `01_crops/report.json`: nguồn ROI, landmark có/không, crop transform.
- `02_fit/report.json`: coarse profile joint gate hoặc lỗi fitting.
- `pipeline.report.json`: trạng thái cuối của lần chạy.

Không mặc định lỗi hết sau khi FaceBox có bbox: crop/fitting/camera vẫn có thể
bị chặn ở bước sau. Chưa chạy detector FaceBox thật trên ảnh right hoặc inference
CUDA tại môi trường tạo gói này, vì không có weights và Torch tương ứng.

## Xác minh

32 test logic (23 cũ +9 mới) qua trên CPU. Kiểm tra bản đồ padding về pixel nguồn,
loại bbox thiếu dữ liệu/score thấp, giữ front bắt buộc landmark, không giả landmark
profile và loại fit sai kích thước/vị trí. Architecture PASS.
Full suite trong runtime hiện tại: baseline223 và bản mới255 cùng2failures,
12errors,47skipped; không thêm tên test lỗi. Runtime đã được dọn nên thiếu nhiều
SDK/tài nguyên hơn lần kiểm tra24.1; không so sánh tổng số test giữa hai runtime
như một phép đo hồi quy. 63checksum runtime V2.3 vẫn khớp.

Các ngưỡng chưa được hiệu chuẩn trên bộ dữ liệu độc lập. Chọn mặt nổi bật bằng
bbox không phải xác minh danh tính, nhất là ảnh có nhiều người.
