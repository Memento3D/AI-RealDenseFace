# Memento3D Figurine V3.1 — tóc nền liên tục và mắt có texture

## Cài đúng vào project hiện tại

Đây là **gói cập nhật cho V3 đang chạy được**, không chứa ảnh, model weights, môi trường hoặc source project đầy đủ. Chép nội dung ZIP trực tiếp vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Sau khi chép phải có các đường dẫn ngay dưới project:

- `RUN-FIGURINE-V31.ps1`
- `figurine-v31-code.json`
- `configs/figurine04.v31.json`
- `scripts/run_figurine_v31.py`
- `scripts/figurine_surface31.py`
- Hai file cập nhật: `scripts/run_figurine_v3.py`, `scripts/figurine_preview.py`.

Chấp nhận thay thế hai file cập nhật của V3. Không tạo thêm thư mục `Memento3D-Figurine-V3` lồng vào project. Không thay đổi `.venv`, `runs`, `input` hoặc model cache.

PowerShell ngay tại project hiện tại:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-FIGURINE-V31.ps1
```

Launcher tự dùng thư mục chứa nó làm project root, kiểm tra checksum code để phát hiện chép thiếu/lẫn phiên bản, đọc workspace/model từ profile, tạo run mới và mở `review.html`. `ExecutionPolicy Bypass` chỉ áp dụng tiến trình PowerShell này, không chỉnh chính sách lưu trên máy.

Không cần cài thêm dependencies nếu V3 vừa chạy thành công. Nếu dùng môi trường mới, launcher có tùy chọn `-InstallDependencies`, dùng requirements CPU hiện có của V3. Không cài vào conda/GPU venv.

Có thể chạy Python trực tiếp:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_figurine_v31.py --profile configs\figurine04.v31.json
```

Nếu workspace đã chuyển chỗ:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-FIGURINE-V31.ps1 -Workspace "D:\duong-dan\head.workspace.html"
```

## Đổi logic nào?

**Tóc:** dùng tia giao với da đầu nguồn để lấy bề mặt nền, làm mượt trong tọa độ quanh đầu, dựng một lớp cap kín có đường chân tóc liên tục. Mask ba ảnh chỉ giúp chỉnh đường biên. Vùng ngoài ảnh, khuất hoặc thiếu depth không cắt thủng cap. Phần gáy chưa có camera ảnh sau dùng đường biên thiết kế có giới hạn. Lọn lớn/rãnh nông đi theo đường cong, giảm dần ở đỉnh, mép tóc và gáy. Không voxel hóa nền tóc hoặc khuôn mặt.

**Mắt:** thay màu theo đỉnh bằng texture 1024x512 cho mỗi mắt, có chuyển biên mềm. Ước lượng vùng tối trong mắt ảnh trước chỉ là một ứng viên; hướng cuối ưu tiên 75% về camera chung để giảm sai lệch hai bên. Trường hợp không đủ hỗ trợ sẽ dùng hướng camera và ghi rõ fallback. Không có model đo gaze/iris chuyên dụng hoặc bảo đảm ánh nhìn đúng người. Mí mắt và đầu nguồn giữ nguyên.

**Miệng:** bản V3.1 mặc định chỉ xuất biểu cảm nguồn. Nụ cười thử vẫn có thể bật bằng `smile.enabled`, giữ nguyên cảnh báo là biểu cảm giả lập.

**In:** giữ surface boolean của V3; đóng biên cổ/miệng trong, hợp nhất thành khối đặc và lấp khoang kín. STL là mm/+Z lên; GLB là m/+Y lên. Không xuất thành công nếu mesh rời hoặc hở.

## Custom không phải sửa code

Chỉnh `configs/figurine04.v31.json`; profile không bị khóa bằng checksum code:

- `hair.base_mm`: độ dày nền theo hướng xuyên tâm; không phải kết quả đo minimum-wall.
- `hair.ridge_mm`, `groove_mm`: độ nổi lọn và độ sâu rãnh; rãnh không được vượt nửa độ dày nền cấu hình.
- `hair.clumps`, `sweep`, `part`: số lọn, độ cong và lệch hướng lọn; không phải điểm ngôi tóc đã đo từ ảnh.
- `eyes.iris_ratio`, `pupil_ratio`, `highlight_ratio`: thiết kế màu trong phạm vi cầu mắt gốc.
- `eyes.per_eye_yaw_deg`, `per_eye_pitch_deg`: hai phần tử tương ứng mắt ModelX âm và ModelX dương; chưa xác nhận quy đổi trái/phải giải phẫu.
- `head_neck_height_mm`: kích thước thử đầu kèm cổ, mặc định 35mm. Các độ dày tóc/chi tiết tính mm cần chỉnh theo kích thước thành phẩm.

V3 cũ vẫn chạy qua `run_figurine_v3.py` với `configs/figurine04.json`; không tự chuyển profile cũ sang V3.1.

## Điều còn cần duyệt

Nền mới mượt và liên tục hơn nhưng kiểu tóc vẫn là thiết kế đơn giản từ scalp/mask, không phải bản sculpt hoàn chỉnh hay tóc sau gáy đã được quan sát. Lọn có thể còn quá nhẹ hoặc quá đều theo gu thẩm mỹ. Không thay thế việc đánh giá độ giống của đầu nguồn. Màu trong GLB là hướng dẫn sơn; Saturn resin không in texture màu.

Watertight/one-component không đồng nghĩa đã in thành công. Chưa kiểm tra slicer, supports, kích thước chi tiết tối thiểu, loại resin hoặc in vật lý.

## Cấu hình mềm hơn

Bản mặc định làm lọn rõ hơn. Cấu hình soft có ảnh so sánh sẵn trong demo; xuất riêng bằng:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_figurine_v31.py --profile configs\figurine04.v31.soft.json
```
