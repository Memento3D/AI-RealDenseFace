# Source v0.13.0 · Mask tự động

Bản đầy đủ kế thừa v0.12. Luồng mới tạo mask bằng BiSeNet ResNet18 ONNX, tự chọn màu nền từ nhiều pixel da rồi tạo workspace. Không bắt buộc tô mask bằng tay; editor vẫn có để sửa ngoại lệ. Chưa phải pipeline production đã được đánh giá chất lượng.

## 1. Cập nhật source

Giải nén nội dung gói vào dự án hiện có, cập nhật source trùng tên. Giữ nguyên môi trường, vendor, weights, input, runs và cấu hình cục bộ. Không chạy lại fit đầu.

## 2. Setup một lần

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/face-parsing.cpu.txt

.\.venv\Scripts\python.exe .\scripts\setup_face_parser.py `
    --output D:/AI-cache/face-parsing/resnet18.onnx
```

Model tải từ release chính thức `yakhyo/face-parsing`, khoảng 51 MiB. SHA-256 được cố định trong code và kiểm tra trước/sau inference. Không cài Torch hoặc thay CUDA. Script setup không ghi đè model có hash khác; không tải ảnh người dùng lên mạng.

## 3. Chạy tự động

```powershell
.\.venv\Scripts\python.exe .\scripts\optimize_head.py `
    --head-run runs/rdf_open_20261023_152718_f3ce74/three_photo_7cf7011b `
    --texture-run runs/rdf_open_20261023_152718_f3ce74/three_photo_7cf7011b/three_textures_73087dd7 `
    --auto-mask `
    --face-parser-model D:/AI-cache/face-parsing/resnet18.onnx
```

Mỗi ảnh chạy lần lượt với ONNX Runtime CPU. Crop vuông suy ra từ phép chiếu đầu đã có, resize 512×512, normalize RGB ImageNet, inverse crop về đúng pixel ảnh gốc. Giữ da, lông mày, mắt, tai, mũi, miệng, môi và cổ; loại nền, tóc, mũ, quần áo, kính, khuyên tai và vòng cổ theo nhãn model. Pixel độ tin cậy thấp bị loại; ngưỡng 0.55 là cấu hình thử nghiệm chưa hiệu chuẩn. Mask được co 1 pixel để giảm lem biên.

Màu nền lấy median nhiều pixel da đủ tin cậy, loại phần sáng/tối cực đoan. Đây vẫn là ước lượng màu từ ảnh có ánh sáng, không phải albedo thật.

## 4. Output

Terminal in đường dẫn workspace mới và `auto_masks.json`. Mở đúng `head.workspace.html` được in ra.

Trong `auto_masks_<id>`:

- `front/left/right.mask_review.jpg`: vùng đỏ bị loại.
- `front/left/right.mask.png`: mask trắng giữ, đen loại.
- `front/left/right.labels.png`: ảnh chỉ số lớp, không phải ảnh RGB.
- `auto_masks.json`: mask và màu nền ràng buộc với viewer nguồn.
- `report.json`: class counts, crop, model hash, thời gian và các cảnh báo.

Trong workspace:

- Mask hiển thị **tự động, chưa review**. Không tự gán thành người đã duyệt.
- Vẫn có trước–sau, cân màu, nguồn ảnh và màu nền.
- Chỉ mở editor nếu thấy lỗi. Editor nạp sẵn mask tự động. Sau khi sửa, tải JSON rồi dùng `--masks-json` thay cho `--auto-mask`.

## Giới hạn cần giữ rõ

- Model không có lớp tai nghe riêng; có thể nhầm tai nghe thành tai hoặc nền. Với luồng chụp MVP, yêu cầu tháo phụ kiện che mặt/tai.
- Phân vùng không phải kiểm chứng mask đúng. Mask tóc mái, tai và góc bên có thể cần sửa.
- Cảnh báo hiện có chỉ gồm heuristic diện tích da, thiếu vùng mắt chính diện, kính và thiếu mẫu màu. Không dùng làm score chất lượng hoặc cổng tự động chấp nhận production.
- Chưa tự sửa mắt/miệng, ghép tóc/mũ, bake UV atlas hay làm đầu kín để in.
- Khi triển khai thật cần đánh giá nhiều người/góc/ánh sáng và thiết kế bước chụp lại hoặc review theo kết quả, không phụ thuộc vào model luôn đúng.

## Nguồn và điều kiện sử dụng

- Code tham khảo: https://github.com/yakhyo/face-parsing tại commit `8a4729d95118d0e97c44185f9bdef3d6bfeaaf99`.
- Model: https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx
- SHA-256: `0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f`.
- Repo công bố code MIT. Dataset CelebAMask-HQ công bố điều kiện nghiên cứu phi thương mại: https://github.com/switchablenorms/CelebAMask-HQ . Không coi giấy phép code là kết luận về quyền thương mại của weights/dữ liệu; cần làm rõ trước khi thương mại hóa. Bản tích hợp này là thử nghiệm nghiên cứu.

## Verification

135 unit tests và architecture check đạt; kiểm tra normalize RGB, decode nhãn, chính sách lớp, inverse crop và ràng buộc mask/nguồn. Đã tải model chính thức, kiểm tra hash và chạy ONNX Runtime CPU thật trên ảnh chính diện gốc 1280×720; output `[1,19,512,512]`. Lần kiểm tra này dùng crop được chỉ định để xác minh inference, chưa phải bằng chứng end-to-end cho bộ đầu/ba ảnh hiện tại của bạn. Workspace với crop suy từ mesh cần được review trên máy bạn.
