# Memento3D V2.3 — sửa UV mắt, giữ chi tiết mí, giới hạn rút khối tóc

Bản này thực hiện các bước sau đối chứng V2.2. Không thay geometry khuôn mặt hoặc nhãn cầu, không khoét hốc, không thêm relief. Tóc vẫn dùng mẫu 21778, chưa phải tái dựng kiểu tóc cá nhân hoàn toàn.

## Chạy trên thư mục dự án hiện tại

Giải nén **nội dung gói mã trực tiếp** vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Launcher `RUN-PIPELINE-V23.ps1` phải nằm ngay trong thư mục dự án, cùng cấp `.venv`, `scripts`, `configs`.

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-PIPELINE-V23.ps1
```

Dùng lại `.venv\Scripts\python.exe`, workspace và parser hiện có. Nếu thiếu dependencies CPU, launcher cài bộ requirements vào venv chính; không thay môi trường WSL/CUDA/RDF. Tạo thư mục `runs\pipeline_v23_<thời gian>` mới và mở review.html, không ghi đè lần chạy cũ.

Workspace mặc định:
`runs/v1_20261025_132459_bbbc53c9/three_photo_346bb6bf/head_workspace_3cdeba95/head.workspace.html`

Parser mặc định: `D:/AI-cache/face-parsing/resnet18.onnx`.

Đường dẫn khác có thể truyền bằng `-Workspace` và `-ParserModel`. Gói không chứa parser hoặc workspace cá nhân. Không cần huấn luyện mô hình mới.

Có thể tăng tốc bằng kết quả V2.1 hợp lệ của đúng workspace/cùng kích thước:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-PIPELINE-V23.ps1 -ReferenceRun ".\runs\pipeline_v21_THOI_GIAN_DA_CHAY"
```

Thay đường dẫn ví dụ bằng thư mục có thật, gồm report.json, checksums.json và compact_21778_0p25/appearance.glb. Mã kiểm tra checksum, chủ thể, scale, asset và kiểm chứng ray. Không cần chạy lại V2.1; không có ReferenceRun thì tự tính correspondence mới. Thời gian thực tế được báo trong gói Results04 dùng đường tăng tốc có kiểm chứng, không phải benchmark Windows/chạy mới hoàn toàn.

## Thay đổi

### Mắt

- Bỏ chuyển nguồn theo từng tam giác eyeball. Mỗi mắt dùng một phép chiếu liên tục từ ảnh trước và một crop gốc, không vẽ lại iris hoặc đoán gaze.
- Các corner trùng vertex phải có UV giống nhau. Nếu nguồn chiếu bị lệch hoặc mắt vượt ảnh, mã từ chối thay vì tự dựng bù.
- Tọa độ, normals và indices geometry mắt/mặt được giữ. Đây là sửa vật liệu/UV, không thay anatomy.
- `appearance.glb` mặc định giữ texture mắt unlit để tránh đánh sáng ảnh nguồn lần hai.
- `eye_pbr_probe.glb` là đối chứng PBR roughness 0.16 trên cùng geometry/texture. Nó có thể có catchlight khi viewer có đèn phù hợp, nhưng ảnh gốc vẫn chứa ánh sáng; không gọi đây là albedo mắt đã hiệu chuẩn. Không dùng preview CPU Lambert để kết luận về specular.

### Da và mí

- Giữ một màu nền tham chiếu từ vùng da sáng đáng tin cậy, áp dụng cho mặt/tai/cổ.
- Thêm residual độ sáng có giới hạn trong dải semantic skin hẹp quanh mắt, không high-pass toàn mặt. Ngoài vùng feature và dải mí, màu nền giữ nguyên.
- Giữ chroma của màu nền trong dải mí; giảm nguy cơ mang quầng màu từ ảnh trở lại. Đây vẫn là heuristic, không bảo đảm loại sạch mọi bóng hoặc phục hồi sắc tố thật.
- Feature phía trước dùng một trường ảnh trước liên tục, tránh đổi nguồn từng tam giác ở vùng này. Atlas head tăng cell từ 16 lên 32; mắt dùng UV crop trực tiếp, không qua atlas tam giác.

### Tóc

- Giữ cavity fit và phương pháp tóc mai V2.2.
- Bước shell vẫn đưa mặt trong vào sọ nhưng giới hạn rút vào theo radial của các vertex proxy khối ngoài: mặc định 1.0 mm so với cage đã fit. Không ép toàn bộ mặt ngoài về body 0.52 mm.
- Budget này kiểm tra trước bước custom tóc mai; vùng chuyển tiếp/rìa không được gọi là phân vùng outer surface chính xác. Không phải ARAP, bảo toàn thể tích hoặc chứng nhận không tự giao.
- Tóc sẽ có thể tích và độ phồng lớn hơn V2.2. Đó là đối chứng giữ phom, không phải khẳng định càng lớn càng đẹp hoặc càng giống ảnh.

## Xem kết quả trước khi chạy

Giải nén gói Results04 riêng, mở review.html. Có ảnh đối chứng V2.2 và cận mắt Cycles để xem ngay, không phải chạy lại.

- `selected/appearance.glb`: bản màu mặc định, mắt unlit nguồn.
- `selected/unlit.glb`: kiểm tra màu nền và UV, không chiếu sáng trên đầu.
- `selected/eye_pbr_probe.glb`: đối chứng vật liệu có phản xạ mắt, không đổi mesh.
- `selected/head_mm.stl`: đầu–tóc hợp nhất, mm/Z up; `hair_mm.stl`: tóc riêng, chưa có cơ cấu tháo lắp.
- `eye_continuity.json`: sai khác màu tại cạnh chung eyeball, cùng vị trí 3D; không phải điểm thẩm mỹ hay độ giống.
- `shared/bright_skin.json`: reference màu, giới hạn chi tiết mí, UV/crop mắt.
- `selected/candidate.json`: tóc trước/sau và các gate; `surface_audit.json` nằm trong thư mục ứng viên.
- `visualcontrol/`: ảnh cận mắt Blender Cycles CPU, dữ liệu snapshot và script tái render.
- `selected/experimental_supports`: support thử nghiệm, chưa cắt lớp máy.

## Giới hạn

Đầu/cổ mặc định 22 mm cho hướng tượng toàn thân 8–10 cm. Chỉ một người và một tóc ngắn đã được kiểm thử. Front-photo UV không phải mô hình gaze đa góc cá nhân hóa. Nguồn ảnh mắt chỉ khoảng 66×65–66 pixel mỗi crop; không thể tạo chi tiết nguồn không có.

Tóc vẫn phụ thuộc dáng mái/viền của template. Giữ khối ngoài không tự bảo đảm hài hòa với từng người. Chưa có đánh giá giống/đẹp độc lập, độ dày chính xác toàn cục, chứng minh không tự giao toàn bộ, hiệu chuẩn resin, slicer/mã máy hoặc in vật lý. `ready_for_machine=false`. Màu GLB không tự xuất hiện trên STL hoặc bản in resin một màu.

Launcher chưa chạy trên Windows trong môi trường kiểm thử này; Python CPU, GLB native import và ảnh Cycles đã được chạy thực tế. Không sửa các bản V2.1/V2.2 hoặc kết quả cũ của bạn.
