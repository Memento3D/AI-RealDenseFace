# Head24.3 — sửa workspace và tiếp tục từ fitting đã lưu

Lỗi `covered[i,:,None]` là lỗi đánh chỉ số NumPy trên list do mã workspace.
Hai bước trước đã exit0 trong lần chạy `head24_20261029_044424_683718`.
Không cần chạy detector, FaceBox, RDF hoặc joint fitting lại để sửa lỗi này.

Giải nén toàn bộ ZIP vào thư mục project hiện tại, đè các file Head24.
Chạy tại thư mục project:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD24.ps1 -ResumeRun ".\runs\head24_20261029_044424_683718"
```

Resume kiểm tra status hai stage trước, status CROPS_PREPARED/FIT_GATES_PASSED,
SHA256 metadata crop, từng ảnh crop và projection_bundle.npz. Từ chối artifact
bị đổi/không đầy đủ. Không lấy latest run hay dùng cache của người khác.

Đầu ra nằm trong `runs/head24_resume_<timestamp>`. Source run và các log lỗi cũ
không ghi đè. Báo cáo mới ghi `resumed_from`, hash fit report và các artifact
đã xác minh dùng lại; chỉ stage03/04 được thực thi. Không giả lập exit0 cho01/02.
Stage03 chạy bằng Python chính trên CPU; stage04 vẫn là quy trình V2.4 hiện tại.
Chưa cần xóa bất kỳ thư mục cũ nào.

Khi chạy bộ ảnh khác từ đầu, vẫn dùng:
```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD24.ps1 -InputDir ".\input\05"
```

## Xác minh bản sửa

- Tách phép tính texture weights để chuẩn hóa list thành array, kiểm shape3views
  và facing hữu hạn. Giữ front priority và hard-assignment semantics.
- Regression từ list bool đúng như dữ liệu runtime gây lỗi; test shape và NaN.
- Test tích hợp CPU thực sự: tạo NPZ lưới synthetic242triangles +3PNG1024,
  chạy visibility, kiểm camera, tính weights, ghi HTML workspace và đọc lại
  payload/camera/geometry. Không mock bước này. Đây là fixture kiểm code,
  không phải tái dựng người thật hay bằng chứng in được.
- Test checksum resume, từ chối crop đổi, fit chưa pass, bảo toàn source files.
- Test orchestration dùng subprocess doubles rõ ràng để xác nhận resume chỉ
  lập lịch workspace và figurine, không detector/GPU fit. Test doubles không
  phải kết quả reconstruction.
- 40 test Head24 qua. Full suite runtime hiện tại263tests:2failures/12errors/
  47skipped đã có ở baseline223; không thêm tên lỗi. Architecture PASS.
- 63 checksum runtime V2.3 không thay đổi. Không sửa vendor/weights/model pins.

Chưa chạy tiếp projection_bundle thật của bộ05 tại đây vì file đó chưa được
upload; chưa khẳng định đầu ra cuối của05 đã đúng/hài hòa/in được. Các gate và
status ready_for_machine=false vẫn giữ nguyên. Xem log trong verification/.
