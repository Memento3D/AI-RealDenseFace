# Memento3D Head API — RDF + hair-cap MVP

MVP hiện chạy **RealDenseFace + cap tóc tham số**, không chạy TripoSR và không có API assembly:

1. Bắt buộc gửi `front`, `left`, `right`; RDF dùng ba ảnh này để fit đầu.
2. Có thể gửi thêm `back`; ảnh này chỉ ràng buộc silhouette/đường tóc phía sau, không tham gia RDF fit.
3. Parser tạo mask tóc; thuật toán fit ranh giới và độ phồng theo các góc nhìn, dùng prior có giới hạn cho vùng không quan sát.
4. Xuất RDF head và hair cap riêng. Cap là solid hình học độc lập; không Boolean-union với đầu.

MVP chưa suy luận hướng tóc/ngôi từ ảnh. Độ dày cap hiện là policy thử nghiệm, scale thiết kế mặc định là **22 mm cho chiều cao mesh RDF**; đây không phải hiệu chuẩn máy in hay kích thước đo từ người. `ready_for_machine=false` là chủ đích.

## Chạy

Nếu API chưa setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/setup/SETUP-HEADAPI.ps1
```

Mở API:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/api/RUN-HEADAPI.ps1
```

API: <http://127.0.0.1:8090/> · Swagger: <http://127.0.0.1:8090/docs>

API cần RDF weights, parser ONNX tại `paths.parser`, và các môi trường trong `configs/headapi.json`. `paths.haircap_python` hiện trỏ tới môi trường geometry đã có sẵn; worker dùng thư viện mesh và parser, **không nạp hoặc gọi model TripoSR**.

## Contract tích hợp Spring BE

Các endpoint nội bộ giữ theo cách `HttpAiGenerationClient` đang gọi: `GET /health`, `POST /jobs`, `GET /jobs/{job_id}` và `GET /jobs/{job_id}/outputs/glb?attempt_id=...`. BE gửi `X-API-Key`; cùng một secret phải được cấu hình riêng ở AI server (`HEADAPI_API_KEY`) và Spring (`app.ai.api-key`, nên lấy từ environment/secret manager). Khi chưa cấu hình key, mọi endpoint ngoại trừ health/docs từ chối request với `503`; khi cấu hình, chúng yêu cầu header hợp lệ. Không commit key hoặc gửi key tới FE.

### Tạo và lưu key trên Windows

1. Tạo key ngẫu nhiên đủ mạnh bằng Python:

   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

   Giữ giá trị này riêng tư; không dán vào chat, Git, ảnh chụp màn hình hay FE.
2. Tạo file local bị Git ignore từ mẫu:

   ```powershell
   Copy-Item .\.env.example .\.env.local
   notepad .\.env.local
   ```

   Điền `HEADAPI_API_KEY=<key vừa tạo>` và giữ `HEADAPI_ALLOWED_IMAGE_HOSTS=res.cloudinary.com`. Nếu Cloudinary của bạn trả URL từ host khác, thêm đúng hostname vào danh sách phân tách bằng dấu phẩy; chỉ thêm host tin cậy.
3. Chạy API trong cùng repo:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts/api/RUN-HEADAPI.ps1
   ```

   Script đọc `.env.local` vào process environment, không in key ra terminal và dừng nếu thiếu key. Kiểm tra file local đang bị ignore:

   ```powershell
   git check-ignore .env.local
   ```

4. Đặt **cùng key** ở Spring trong biến môi trường `AI_API_KEY` (ánh xạ tới `app.ai.api-key`) hoặc secret store của BE. Không gửi key trong body, query string hay cho FE. Trên máy deploy, dùng secret manager/environment injection; không dùng `.env.local` trong production.
5. Khi cần expose qua ngrok, để API chạy ở `127.0.0.1:8090`, mở tunnel `ngrok http 8090`, rồi cấu hình Spring `app.ai.base-url` bằng HTTPS ngrok URL. Dùng HTTPS; thay API key nếu URL/log/config bị lộ. `/health` không đòi key để BE health check, nhưng mọi route khác đều cần `X-API-Key`.

Request BE → AI phải gửi **3 ảnh bắt buộc** (`front`, `left`, `right`), có thể thêm ảnh thứ tư `back`. Mỗi ảnh cần `image_id`, `input_url` và `view_hint`; `job_id` và `attempt_id` là UUID do BE cấp để poll/finalize đúng attempt:

```json
{
  "job_id": "UUID",
  "attempt_id": "UUID",
  "profile": "default",
  "input_images": [
    {
      "image_id": "UUID",
      "input_url": "https://res.cloudinary.com/.../front.jpg",
      "view_hint": "front",
      "sha256": "optional-lowercase-sha256"
    },
    {
      "image_id": "UUID",
      "input_url": "https://res.cloudinary.com/.../left.jpg",
      "view_hint": "left"
    },
    {
      "image_id": "UUID",
      "input_url": "https://res.cloudinary.com/.../right.jpg",
      "view_hint": "right"
    }
  ]
}
```

AI chỉ nhận 3–4 ảnh, `image_id` và `view_hint` phải duy nhất, các hint bắt buộc là `front`, `left`, `right`, và `back` tùy chọn. Payload cũ chỉ có `input_image_id`/`input_url` không còn đáp ứng contract và bị từ chối `422`. FE phải cung cấp nhãn góc; BE chuyển tiếp nhãn đó và không suy đoán góc từ `sort_order`. URL phải là HTTPS trên host trong `HEADAPI_ALLOWED_IMAGE_HOSTS` (mặc định `res.cloudinary.com`); server không theo redirect và giới hạn mỗi file 12 MiB. Nếu gửi `sha256`, đó phải là SHA-256 của bytes ảnh gốc tải về. Không gửi Cloudinary secret cho AI.

Response status tương thích `AiJobResponse`: `queued`, `running`, `done`, `failed`, cùng `job_id`, `attempt_id`, `stage`, `progress`, `error_code`, `error_message` và timestamp. `done` chỉ khi cả head và hair-cap đã xuất thành công. Output là một GLB tại endpoint trên, chứa hai node riêng `RDF_Head` và `Hair_Cap`; không Boolean-union. `hair/scene.manifest.json` mô tả component và hệ tọa độ. Thiếu một trong ba view bắt buộc, hint trùng, ảnh không lấy được hoặc input không hợp lệ sẽ bị từ chối trước khi queue.

Health trả `{"status":"ok", ...}` để tương thích kiểm tra của BE. API local multipart `/v1/head/jobs` vẫn giữ behavior hiện tại, yêu cầu front/left/right và back tùy chọn.

Chụp ảnh cùng người, nền/ánh sáng ổn định, đầu thẳng tương đối:

| Tên file trong input folder | Bắt buộc | Mục đích |
|---|---:|---|
| `front.png` | Có | RDF và hairline phía trước |
| `left.png` | Có | RDF và silhouette bên |
| `right.png` | Có | RDF và silhouette bên còn lại |
| `back.png` | Không | Thêm ràng buộc tóc phía sau; cần thấy hai tai để căn góc |

Gửi job:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/api/TEST-HEADAPI.ps1 -InputDir ".\input\04" -Wait
```

Nếu folder có `back.png`, client tự gửi cả bốn ảnh. Nếu không, chạy chế độ ba ảnh. API từ chối thiếu một trong front/left/right.

## Kết quả và kiểm tra

- Theo dõi job ở trang API hoặc `GET /v1/head/jobs/{job_id}`.
- Thành công của luồng xuất là `status=MVP_EXPORTED`; điều đó **không** có nghĩa là mesh đã in được.
- Tải `rdf/rdf_face_gray.glb` để xem đầu RDF.
- Tải `hair/hair_cap_native.glb` để xem cap ở đơn vị native RDF.
- `hair/hair_cap_design_mm.stl` là cap kín ở scale thiết kế giả định 22 mm chiều cao mesh RDF, chưa hiệu chuẩn với máy in/vật liệu.
- `hair/report.json` ghi rõ ảnh back có được dùng không, mức ranh giới quan sát/suy diễn và kiểm tra topology.
- `hair/front.fit.png`, `left.fit.png`, `right.fit.png` (và `back.fit.png` nếu có) hiển thị mask tóc màu xanh và biên cap dự đoán màu hồng; `photo_consistency` chỉ đo độ khớp với parser/camera hiện tại, không phải đánh giá likeness độc lập.
- Nếu lỗi, xem `errors`, `current_stage` và log tương ứng. RDF fail thì cap không chạy.

Ba ảnh và bốn ảnh được xử lý thành các job riêng để so sánh. Dùng cùng front/left/right và cùng một người; job bốn ảnh chỉ thêm back. So sánh mask overlays, `hair_boundary`, hình cap từ sau và báo cáo. Phần đỉnh đầu vẫn chưa được quan sát trực tiếp.

## Giới hạn MVP

- Segmentation nhận biết vùng tóc; fit hiện dùng mask/silhouette để khớp hairline và volume thô. Nó không tái tạo sợi tóc và chưa tự nhận diện ngôi/hướng mọc.
- Vùng khuất ở chế độ ba ảnh dùng prior; ảnh back chỉ bổ sung quan sát phía sau và cần căn qua hai tai.
- Mesh topology kín được kiểm tra tự động, nhưng self-intersection, độ dày tối thiểu thực tế, clearance/lắp ghép với đầu, vật liệu, slicer và bản in vật lý chưa được xác nhận.
- Không có Boolean union, không khẳng định độ giống/identity hoặc printability.
- Các endpoint và chế độ API assembly/TripoSR cũ đã bị bỏ khỏi luồng Head API. Những job cũ trong `runs/headapi` được giữ nguyên; không assemble lại được bằng API mới.

Sau khi chỉnh code, kiểm tra theo quy ước project:

```powershell
python -m unittest discover -s tests -v
python scripts/check_architecture.py
```
