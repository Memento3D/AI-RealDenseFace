# MVP local: ảnh → hình học phục vụ chuẩn bị in

Ưu tiên làm việc hiện tại: xác nhận khuôn mặt và texture riêng trước, rồi mới
quay lại tóc. Chưa nối luồng production. Lệnh dưới là thử nghiệm hình học đã có,
không phải yêu cầu tiếp tục tối ưu tóc hay thay đổi API ở bước hiện tại.

Mục tiêu hiện tại là hoàn thiện luồng hình học và bằng chứng kiểm tra. Tạm giữ
thuật toán mặt/tóc; không tự động fine-tuning hay áp dụng các refinement đang
thử nghiệm. Kết quả xuất hình học không đồng nghĩa sẵn sàng chạy máy in.

## Chạy một lệnh

Từ thư mục gốc, dùng các môi trường/model local đang cấu hình trong
`configs/headapi.json`:

```powershell
.venv/Scripts/python.exe -m tools.headapi.prepare_print --input input/04 --output runs/experiments/my-preparation-01 --head-height-mm 30
```

Input cần đúng một ảnh `front`, `left`, `right`, đuôi JPEG/PNG/WebP. Lệnh này
không dùng ảnh `back` hoặc ảnh trong `references`. Không tự cài dependencies,
tải weights hoặc đổi model pins. Mỗi lần phải dùng output mới; không ghi đè run.

Luồng đang thực thi: chuẩn hóa ảnh → phát hiện/crop → RDF ba góc → xuất mặt →
fit hair cap → đóng biên đầu → hợp nhất đầu/mắt/tóc → cắt cổ loe theo policy
thử nghiệm → gắn đế → xuất STL/GLB mm, Z-up → đọc lại kiểm tra → lấy mẫu vùng mảnh.

Kích thước 30mm là chiều cao đầu+cổ trước cắt cổ/gắn đế, không phải chiều cao
toàn sản phẩm, không phải kích thước giải phẫu được khôi phục từ ảnh. Policy
hiện cắt 20% chiều cao từ dưới và thêm đế 3mm; đây là giả định thử nghiệm cần
kiểm tra theo mẫu, chưa phù hợp mọi loại cổ/đầu. Model đặc, chưa hollow.

## Artifacts và trạng thái

- `head_hair_pedestal_mm.stl`, `.glb`: hình học thật đã được xuất.
- `manifest.json`: cấu hình, checksum ảnh/code/artifact, lệnh từng stage, lỗi,
  kết quả topology và vị trí cảnh báo chord ngắn.
- `rdf/`, `hair/`, `rdf_crops/`, `inputs/`: bằng chứng trung gian của đúng attempt.
- Các file `.log`: lỗi và thông tin từng stage.

`FAILED`: stage hoặc validation thất bại; không trả giả thành công.

`GEOMETRY_EXPORTED_REQUIRES_REVIEW`: xuất được STL kín, thể tích dương, một
thành phần, winding nhất quán; vẫn còn yêu cầu kiểm chứng. Đây chưa phải trạng
thái `prepared printing` hoàn chỉnh. `ready_for_machine` luôn false trong phiên
bản này; không dùng trạng thái này để tự tạo lệnh in trong BE.

## Kết quả chạy thật

Attempt: `runs/experiments/input04-mvp-preparation-01`.
Kích thước cuối khoảng 17,35 × 23,41 × 27,24mm; khối kín, một thành phần,
thể tích dương. Lấy mẫu 3.000 điểm phát hiện 45 chord dưới ngưỡng thử 0,8mm.
Đó không phải phép đo độ dày tối thiểu hoặc kết luận vật liệu sẽ gãy.

Rà soát riêng trên bản refined trước đó:
`runs/experiments/input04-thin-triage-01/report.json`.
35/45 vị trí có ít nhất 3/5 ray lân cận vẫn ngắn và normal đầu–cuối tương đối
đối nhau; 10/45 nhạy với mép/góc. Không tự động loại bỏ cảnh báo hoặc tăng dày
khuôn mặt. `refined_warnings.glb` hiển thị marker đỏ/cam để kiểm tra vị trí.
Các marker chỉ là annotation, không phải phần model dùng để in.

## Những việc còn thiếu trước khi đem thử in

1. Xác minh các cảnh báo dai dẳng, sửa có chọn lọc và đo thay đổi mặt.
2. Kiểm tra giao cắt toàn cục và độ dày bằng phương pháp phù hợp hơn lấy mẫu.
3. Dùng slicer thật cho Saturn 4 Ultra 16K: orientation, support, island,
   profile resin và file xuất máy. Mặt cắt Trimesh không thay thế slicer.
4. Thử in vật lý khi có máy, ghi nhận resin/exposure/hậu xử lý và kết quả.

Không tuyên bố độ giống, sức bền, in thành công hay calibration sản xuất từ
mesh kín. Sau khi mốc chuẩn bị in ổn định, quay lại cải thiện mặt/tóc và chạy
lại cùng bộ kiểm tra trên từng version.
