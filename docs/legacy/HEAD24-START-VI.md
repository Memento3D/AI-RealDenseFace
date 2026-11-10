# Bản hiện tại: Head24.4

Đọc `HEAD24-4-START-VI.md` trước. Các tài liệu bên dưới giữ lịch sử; số liệu và phạm vi gate hiện tại được giải thích trong tài liệu24.4.

---

# Bản hiện tại: Head24.3

Ưu tiên `HEAD24-3-RESUME-VI.md` để tiếp tục từ fitting đã lưu, không chạy lại GPU. Tài liệu bên dưới giữ lịch sử24/24.1/24.2.

---

# Bản hiện tại: Head24.2

Đọc `HEAD24-2-PROFILE-VI.md` cho cách chạy, nhánh bbox ảnh bên và kết quả test hiện tại. Các ghi chú24/24.1 bên dưới được giữ làm lịch sử; báo cáo runtime cũ không áp dụng cho lần kiểm tra24.2.

---

# Bản sửa 24.1 — tìm mặt trong ảnh toàn thân

Giải nén đè các file Head24 của gói này vào project. Giữ nguyên V2.3/vendor/model.
Chạy lại cùng lệnh RUN-HEAD24.ps1 -InputDir ".\input\05".

Khác bản 24 đầu tiên: quét ảnh gốc và các vùng chồng lấn ở hai cỡ, đưa mọi
landmark về pixel ảnh gốc rồi gộp phát hiện trùng. Quét thêm biến thể nâng sáng
chỉ cho detector; crop/texture cuối vẫn lấy pixel gốc, không đổi màu da.
Không hạ ngưỡng 64 native pixels hay ngưỡng camera 5px. Không dùng bbox giả.
Nhiều cửa sổ làm bước CPU tìm mặt lâu hơn; số calls được ghi trong báo cáo.

Khi dừng, xem `01_crops/report.json` và `<view>.detections.jpg`. Báo cáo phân biệt
NO_FACE_DETECTED và NO_FACE_MEETS_NATIVE_SIZE. DETECTION_UNRESOLVED chỉ có nghĩa
chưa định vị được mặt; không phải bằng chứng ảnh chắc chắn không dùng được.
Các hộp trên ảnh diagnostics là phát hiện detector, chưa phải xác nhận đúng người.
23 test toán/chính sách qua; chưa chạy detector thật hoặc inference GPU mới trên05.

---

# Head24 / Pipeline V2.4 — thử nghiệm ảnh toàn thân sang đầu có kiểm tra căn chỉnh

Đây là gói bổ sung trên V2.3 bạn đã chạy. Giữ nguyên file cũ và vendor;
không sửa .venv, .venv-rdf, WSL, cache OCR hoặc weights. Không có model/ảnh/run
trong gói code. Chưa chạy được end-to-end GPU/Windows ở môi trường tác giả.

## Cài và chạy

Giải nén nội dung ZIP trực tiếp vào:
`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Cần thấy `RUN-HEAD24.ps1` cạnh `.venv`, `scripts`, `configs`, `src`.
Không giải nén thành thư mục con mới. Giữ nguyên assets tóc đã có.

Chạy bộ 05:
```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD24.ps1 -InputDir ".\input\05"
```

Một ảnh front/left/right duy nhất cho mỗi góc; .png/.jpg/.jpeg đều được.
Bộ 04 khi muốn đối chứng: đổi `05` thành `04`. Không truyền workspace/cache04.
Kiểm tra file điều kiện trước mà chưa inference:
```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD24.ps1 -CheckOnly
```
CheckOnly xác nhận file/code pin, không chứng minh SDK, CUDA hay inference hoạt động.

## Điều kiện đang dùng

- `.venv\Scripts\python.exe`: bộ CPU V2.3 và MediaPipe (cùng API adapter cũ).
- `.venv-rdf\Scripts\python.exe`: môi trường RDF/CUDA đang có.
- `D:\AI-cache\mediapipe\face_landmarker.task`, SHA256
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
- Parser ONNX cũ ở `D:\AI-cache\face-parsing\resnet18.onnx`.
- Vendor RDF Python/config khớp 48 file bạn vừa gửi; không chỉnh vendor để bỏ gate.
- FLAME2023/assets/FaceBox giữ các SHA đã đo trong run05. Trọng số vitb lấy đúng
  đường dẫn cũ và ghi SHA thực tế; chưa có checksum chuẩn độc lập cho vitb.

Không tự cài hoặc tải lại model. Nếu thiếu mediapipe/task, dừng và gửi lỗi;
không đổi model hash hay cài chồng vào .venv-rdf để thử cho qua.

## Luồng và kết quả

Mỗi lần tạo `runs/head24_<timestamp>` mới:
1. `01_crops`: chọn mặt nổi bật, từ chối nhiều ứng viên tương đương, lưu ảnh crop
   1024, landmark và ma trận thuận/nghịch trên ảnh đã chuẩn hóa EXIF.
2. `02_fit`: truyền bbox landmark trực tiếp cho inferencer. Không gọi nhánh
   tự tìm bbox của RDF, vì nhánh đó có fallback bbox toàn ảnh khi detector thất bại.
   Fitter độc lập 3 crop để lấy FOV riêng; joint identity với layout T=3,C=1.
   So tâm hai mắt 3D với MediaPipe ở front trước và sau joint.
3. `03_workspace`: lưu camera pixel, hình học biểu cảm riêng từng góc và q texture;
   xác nhận P_i X_i khớp q_i. Không tái suy camera từ hình học front.
4. `04_figurine`: dùng da/mắt/tóc khối của V2.3, camera explicit. Tóc vẫn mẫu
   ngắn 21778; không phải bộ tự chọn kiểu tóc hay dựng toàn thân.

Nếu xuất được, mở `04_figurine/review.html`. Màu GLB không tự trở thành màu resin.
Nếu bị chặn, xem `pipeline.report.json`, log stage, `01_crops/report.json`,
`02_fit/report.json`, hoặc `04_figurine/rejection.json`. Không chạy tiếp bằng
workspace cũ và không sửa ngưỡng/checksum để ép xuất.

## Các gate là gì, chưa là gì

- Chọn người: diện tích mặt với prior yếu theo tâm ngang; không phải face ID.
  Nếu người nền chiếm ưu thế, vẫn có rủi ro chọn sai. Ứng viên thứ hai >60%
  điểm cao nhất bị từ chối. Chưa đối sánh danh tính giữa các góc.
- Mặt có ít nhất 64 pixel ở mỗi chiều trong ảnh gốc. Upscale crop không tạo
  chi tiết mới. Ngược sáng chỉ được cảnh báo, không giả lập màu da thật.
- Gate tâm mắt: sai số tối đa 0.25 lần khoảng cách mắt, tỉ lệ khoảng cách
  0.65..1.4. Đây là heuristic chặn sai lệch lớn, chưa phải kiểm định đầy đủ
  môi/mũi, cảm xúc, iris/gaze hay độ giống. Kính vẫn che dữ liệu mắt.
- Sai số khớp contract camera tối đa 0.05px là kiểm tra nhất quán dữ liệu,
  không phải hiệu chuẩn camera hay độ chính xác so với ảnh người thật.
- Tóc dùng hình học front làm proxy ở các góc khác; P95 biến dạng vùng đầu trên
  giữa front và biểu cảm góc bên vẫn bị chặn ở 5px. Không tăng gate V2.3.
- Có thể dừng NEEDS_RETAKE/INPUT_GATE_REJECTED thay vì xuất một mesh sai.
- Chưa có bằng chứng bản mới dựng đúng bộ05. Cần chạy GPU trên máy bạn trước
  khi kết luận đã sửa thành công dữ liệu thật. Không có placeholder kết quả.

## Thay đổi model cần biết

Luồng cũ front dùng vitb, joint dùng vits. Nhánh Head24 dùng vitb cho cả ba
crop để thống nhất đầu vào joint. Vẫn dùng weights vitb đã có, không tải mới;
đây là thay đổi có chủ đích, không phải benchmark tốc độ/VRAM trên RTX4050.

## Kiểm thử

23 test mới qua trên CPU: chọn/loại mặt, crop invertible, sai căn mắt,
rigid transform, camera riêng, quét cửa sổ, gộp trùng và giữ pixel gốc.
Full suite ở cùng runtime hiện tại: baseline296 và bản mới319 cùng 1 failure,
6 errors; không có thêm tên test lỗi. Runtime hiện tại có OpenCV nên số test
chạy được khác log của bản24 trước. Các lỗi còn lại cần tài nguyên/SDK ngoài
ZIP source. Architecture PASS. Chưa chạy FaceLandmarker thật hay inference
GPU mới trên05. Log kiểm tra trước/sau nằm trong verification/.
