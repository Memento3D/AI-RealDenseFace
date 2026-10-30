# Memento3D Head API — MVP v0.2

Front bắt buộc, left/right tùy chọn. `mode=mvp` chạy ảnh → RDF face → TripoSR head → căn mốc mặt → tách tóc → một GLB gồm đầu và tóc. Không nhân bản front thành ảnh bên. RDF dùng các góc có thật; TripoSR chỉ dùng front.

## Nâng cấp bản v0.1 đang chạy

1. Đợi job cũ hoàn thành; Ctrl+C để dừng server.
2. Giải nén gói cập nhật vào gốc project. Giữ nguyên `runs`, các env và cache model. Không cần cài lại Torch.
3. `configs/headapi.json` thêm `paths.parser`, mặc định `D:/AI-cache/face-parsing/resnet18.onnx`. Model này đã được dùng trong các bản HeadLab trước; phải khớp SHA trong module fusion. Nếu bạn đã tùy chỉnh đường dẫn, giữ lại các giá trị đó và thêm parser.
4. Khởi động lại:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADAPI.ps1
```

Mở http://127.0.0.1:8090/ hoặc http://127.0.0.1:8090/docs. API mới hiện version **0.2.0**. `/health` có thêm backend `mvp`; `FILES_PRESENT_NOT_GPU_VALIDATION` chỉ là đủ đường dẫn/file, chưa chạy model.

Nếu chưa từng cài API: chạy `SETUP-HEADAPI.ps1` trước. Setup chỉ tạo env `.venv-headapi` nhẹ; giữ nguyên `.venv`, `.venv-rdf`, `.venv-triposr`. API cần các môi trường/model của bản trước đã setup.

## Ghép ngay từ job bạn đã có — không dựng lại hai đầu

Trong Swagger gọi:

```text
POST /v1/head/jobs/{job_id}/assemble
job_id = d882600a5fe94a5daf8253631df8f2cf
```

Không upload ảnh lại, không cần body. Response HTTP 202 trả **job_id mới** cùng `parent_job_id`. Dùng job_id mới để theo dõi/tải file. Job nguồn và file nguồn được giữ nguyên; child ghi output mới. Nếu nguồn còn RUNNING hoặc thiếu thành phần, API báo lỗi, không giả kết quả.

Hoặc trong PowerShell thứ hai:

```powershell
powershell -ExecutionPolicy Bypass -File .\ASSEMBLE-HEADAPI.ps1 -JobId "d882600a5fe94a5daf8253631df8f2cf" -Wait
```

Giao diện upload có nút **Ghép MVP từ job này** cạnh ô Job ID; điền ID nguồn đã hoàn thành rồi bấm nút.

## Test một lượt mới từ đầu

`POST /v1/head/jobs`, multipart/form-data:

| Trường | Giá trị |
|---|---|
| front | Ảnh JPEG/PNG/WebP bắt buộc |
| left | Tùy chọn |
| right | Tùy chọn |
| mode | **mvp** mặc định; components = hai thành phần riêng; raw = chỉ TripoSR/front |
| resolution | 128 mặc định; 192 hoặc 256 tùy chọn |

Bỏ chọn Send empty value cho file bên không gửi trong Swagger. Chỉ POST một lần; HTTP 202 là nhận job, không phải hoàn thành.

```powershell
powershell -ExecutionPolicy Bypass -File .\TEST-HEADAPI.ps1 -InputDir ".\input\02" -Mode mvp -Views all -Wait
```

Đổi `02` thành `01`–`06`; `-Views` hỗ trợ front/front-left/front-right/all. Các slot bên không có trong folder thì không gửi. Ảnh trong mỗi slot phải thuộc cùng người; hệ thống chưa kiểm chứng danh tính giữa ảnh.

## Theo dõi và kết quả

- `GET /v1/head/jobs/{job_id}`: trạng thái, bước, errors và artifacts.
- `GET /v1/head/jobs/{job_id}/logs?stage=08_mvp_assembly`: log. Bỏ stage để xem bước hiện tại khi RUNNING.
- `GET /v1/head/jobs/{job_id}/files/{relative}`: tên relative lấy trong artifacts, ví dụ **fusion/mvp_head.glb** hoặc **fusion/review.html**.

Các bước mới: 06_mvp_render → 07_mvp_landmarks → 08_mvp_assembly. Tất cả chạy CPU bằng model landmark/parser đang có, không chạy lại neural RDF/TripoSR khi ghép từ job cũ. Không có performance cam kết cho Windows.

Thành công: `status=MVP_ASSEMBLED`, `hybrid_assembly_performed=true`, `rdf_geometry_unchanged=true`. Một GLB có hai node: `RDF_Head_Preserved` và `TripoSR_Hair_Surface`. Mặt giữ geometry của bundle; tóc được transform và chỉnh clearance phía tóc. Đây là **scene assembly**, chưa Boolean union.

File trong `runs/headapi/<job_moi>/fusion`:

- `mvp_head.glb`: đầu và tóc cùng hệ tọa độ RDF.
- `hair_aligned.glb`: tóc riêng đã căn.
- `mvp_color.png`, `mvp_gray.png`, `review.html`: preview.
- `assembly.report.json`: transform, source hashes, residual, thay đổi tóc và giới hạn.

Nếu một bước ghép thất bại, lượt mới có thể PARTIAL, hoặc job ghép riêng FAILED; đọc log thay vì gọi đó là thành công. Các file thành phần vẫn giữ. Hàng đợi chạy tuần tự, tối đa 3 job đang chờ/chạy. Đợi DocumentAI/GPU task khác xong trước khi tạo lượt dựng mới. API không điều khiển các project khác.

## Cách ghép và giới hạn MVP

- Raster có z-buffer → MediaPipe mốc mặt trên render TripoSR → đối chiếu surface 3D với mốc ảnh RDF → similarity dương, xoay/dịch/tỷ lệ đồng nhất, có robust weights. Không kéo méo mặt RDF.
- Parser BiSeNet ONNX chọn tóc trên bốn render, bảo vệ vùng mặt ở góc trước. Region growth theo màu nối từ seed tóc. Đỉnh đầu có bước hoàn thiện vùng suy diễn dựa vào geometry TripoSR và độ cao tương đối với mắt; **không phải quan sát tóc thật ở vùng khuất**. MVP này ưu tiên thử tóc ngắn; tóc dài/buộc, tóc sáng, che mặt hoặc góc khó chưa được kiểm chứng.
- Chỉnh phần tóc ra ngoài convex envelope của đầu để hạn chế bị chìm. Đây là clearance heuristic, chưa chứng minh không giao xuyên hoặc không hở. Một số mảng tóc có thể dày/cứng/không cân đối.
- Da nền chọn band sáng không cực trị trong vùng da semantic, ưu tiên view sáng hơn; giữ màu vùng mắt/mày/môi từ ảnh. Không tuyên bố khôi phục albedo hoặc độ giống khuôn mặt.
- **Hair là surface, đầu vẫn có thể hở cổ. Không phải một solid hợp nhất, chưa có STL/3MF sẵn sàng in**, chưa có toàn thân/đế. `ready_for_machine=false`, `face_identity_verified=false` vẫn đúng.
- Camera debug mỗi nhánh vẫn riêng; chỉ asset MVP được transform chung. Scale xuất native RDF, chưa hiệu chuẩn mm.

## Kiểm chứng

Xem verification/headapi-release.json. Có replay CPU trên ZIP job thực tế và test API với subprocess thật cho bước ghép, không phải mock kết quả dựng. Synthetic tests tách riêng. Chưa chạy API v0.2/PowerShell trên Windows ở môi trường phát hành và chưa kiểm chứng vật lý/in. Trọng số model không kèm trong ZIP code và không đổi pin.
