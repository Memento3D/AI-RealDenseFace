> Tham khảo API của nhánh head/cap trước đây. Luồng demo assembly và setup hiện tại được mô tả trong README ở gốc; thông tin policy ở đây không thay thế cấu hình đang chọn.

# Memento3D Head API — hướng dẫn sử dụng

Các launcher được tổ chức trong [`scripts/`](scripts/README.md); lệnh demo quen thuộc `./START-HEADAPI-ASSEMBLY.ps1` vẫn được giữ ở gốc. Xem [cấu trúc project](docs/architecture/PROJECT-LAYOUT-VI.md), [danh mục tài liệu](docs/README.md) và [runbook demo](docs/guides/WEB-API-DEMO-RUNBOOK-VI.md).

Tài liệu này mô tả API đang xuất hiện trong Swagger của project. API chạy trong cùng một FastAPI service, nhưng có hai nhóm endpoint với mục đích và định dạng input khác nhau:

1. **Backend API** (`/jobs`): BE gửi JSON chứa URL ảnh, API tải ảnh và tạo job; dùng để tích hợp Spring BE.
2. **Local test API** (`/v1/head/jobs`): người phát triển upload file ảnh trực tiếp; phù hợp để thử pipeline trên máy local.

API hiện chạy pipeline thử nghiệm **RDF + hair cap tham số**. Đây không phải Hunyuan/TripoSR, không có API ghép/assembly riêng, và kết quả chưa được xác nhận về độ giống, hiệu chuẩn sản xuất hay khả năng in.

## 1. Khởi động API

Mở PowerShell tại thư mục gốc project. Nếu chưa cài môi trường:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/setup/SETUP-HEADAPI.ps1
```

Tạo `.env.local` từ file mẫu nếu chưa có, sau đó điền API key cá nhân:

```powershell
Copy-Item .\.env.example .\.env.local
notepad .\.env.local
```

Giữ `HEADAPI_API_KEY` bí mật và không commit `.env.local`. Khởi động service:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/api/RUN-HEADAPI.ps1
```

Mặc định:

- Trang giới thiệu: <http://127.0.0.1:8090/>
- Swagger UI: <http://127.0.0.1:8090/docs>
- OpenAPI JSON: <http://127.0.0.1:8090/openapi.json>
- Health check: <http://127.0.0.1:8090/health>

Giữ cửa sổ PowerShell đang chạy server mở trong lúc gọi API. Có thể đổi port bằng `.\scripts/api/RUN-HEADAPI.ps1 -Port 8091`.
Trang `/` cũng nằm sau middleware xác thực; dùng `/docs` hoặc `/health` để truy cập không cần key.

## 2. Xác thực và cách thử Swagger

`GET /health`, Swagger UI, OpenAPI JSON và ReDoc không cần key. Các endpoint xử lý job, status, logs và artifacts cần header:

```text
X-API-Key: <giá trị HEADAPI_API_KEY>
```

Key trên AI server nằm trong `.env.local`. Nếu chưa cấu hình key, endpoint được bảo vệ trả `503`; nếu thiếu/sai key, trả `401`.

### Thử nhanh bằng Swagger UI

1. Mở <http://127.0.0.1:8090/docs>.
2. Chọn `GET /health` → **Try it out** → **Execute**. Response có `status`, `service_status`, `backend_ready` và `ready_for_machine`.
3. Với các route job, Swagger hiển thị danh sách, mô tả và một số tham số. Tuy nhiên, giao diện hiện tại chưa hỗ trợ đầy đủ việc tạo job:
   - `POST /jobs` nhận JSON bằng cách đọc request thô nên Swagger không dựng ô nhập request body.
   - Các route `/v1/head/...` được bảo vệ bằng middleware nhưng không khai báo API key trong OpenAPI, nên Swagger không có ô nhập key cho chúng.
4. Vì vậy, dùng Swagger để xem tài liệu/health; dùng script hoặc lệnh PowerShell ở dưới để gửi job thật.

## 3. Danh sách endpoint trong Swagger

### Endpoint chung

| Method và route | Xác thực | Tác dụng | Kết quả thường gặp |
|---|---|---|---|
| `GET /health` | Không | Kiểm tra API đang chạy và kiểm tra sơ bộ các prerequisite của pipeline. Không chạy reconstruction. | JSON gồm trạng thái service và `backend_ready`. |

### Backend API — BE gửi URL ảnh

| Method và route | Xác thực | Tác dụng |
|---|---|---|
| `POST /jobs` | Bắt buộc | Tạo job từ JSON. AI tải ảnh HTTPS từ host cho phép (mặc định `res.cloudinary.com`) rồi đưa vào queue. Trả `202 Accepted`. |
| `GET /jobs/{job_id}` | Bắt buộc | Lấy trạng thái job theo ID. |
| `GET /jobs/{job_id}/outputs/glb?attempt_id={attempt_id}` | Bắt buộc | Tải GLB khi job đã hoàn thành thành công và `attempt_id` khớp. |

`POST /jobs` yêu cầu `job_id`, `attempt_id` là UUID và 3 ảnh `front`, `left`, `right`; `back` tùy chọn. Mỗi ảnh cần `image_id`, `input_url`, `view_hint`; `sha256` là tùy chọn:

```json
{
  "job_id": "11111111-1111-4111-8111-111111111111",
  "attempt_id": "22222222-2222-4222-8222-222222222222",
  "profile": "default",
  "input_images": [
    {
      "image_id": "33333333-3333-4333-8333-333333333333",
      "input_url": "https://res.cloudinary.com/example/image/upload/front.jpg",
      "view_hint": "front"
    },
    {
      "image_id": "44444444-4444-4444-8444-444444444444",
      "input_url": "https://res.cloudinary.com/example/image/upload/left.jpg",
      "view_hint": "left"
    },
    {
      "image_id": "55555555-5555-4555-8555-555555555555",
      "input_url": "https://res.cloudinary.com/example/image/upload/right.jpg",
      "view_hint": "right"
    }
  ]
}
```

Thay các URL ví dụ bằng URL ảnh thật mà server có thể truy cập. Ảnh phải là JPEG/PNG/WebP, tối đa 12 MiB mỗi ảnh, tối đa 24 megapixel và chiều ngắn ít nhất 64 pixel. URL phải dùng HTTPS, đúng host allowlist, không redirect; các ảnh phải khác nhau. Không gửi Cloudinary secret trong URL hoặc request.

Ví dụ gửi request từ PowerShell (đặt key an toàn trong biến môi trường của terminal trước khi chạy):

```powershell
$headers = @{ "X-API-Key" = $env:HEADAPI_API_KEY }
$body = @{
  job_id = (New-Guid).Guid
  attempt_id = (New-Guid).Guid
  profile = "default"
  input_images = @(
    @{ image_id = (New-Guid).Guid; input_url = "https://res.cloudinary.com/example/image/upload/front.jpg"; view_hint = "front" },
    @{ image_id = (New-Guid).Guid; input_url = "https://res.cloudinary.com/example/image/upload/left.jpg"; view_hint = "left" },
    @{ image_id = (New-Guid).Guid; input_url = "https://res.cloudinary.com/example/image/upload/right.jpg"; view_hint = "right" }
  )
} | ConvertTo-Json -Depth 5

$job = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8090/jobs" `
  -Headers $headers -ContentType "application/json" -Body $body
$job
```

Nếu cần dùng key từ `.env.local`, nạp nó vào biến môi trường của terminal hiện tại mà không in giá trị ra màn hình:

```powershell
$keyLine = Get-Content .\.env.local | Where-Object { $_ -match '^HEADAPI_API_KEY=' } | Select-Object -First 1
$env:HEADAPI_API_KEY = $keyLine.Substring('HEADAPI_API_KEY='.Length).Trim().Trim('"').Trim("'")
Remove-Variable keyLine
```

Giữ lại `job_id` và `attempt_id` từ request. Poll trạng thái:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8090/jobs/$($job.job_id)" -Headers $headers
```

Khi response có `status = "done"`, tải GLB:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8090/jobs/$($job.job_id)/outputs/glb?attempt_id=$($job.attempt_id)" `
  -Headers $headers -OutFile ".\scene.glb"
```

Response status backend dùng `queued`, `running`, `done`, `failed`. Các field chính gồm `job_id`, `attempt_id`, `stage`, `progress`, `error_code`, `error_message` và timestamp. `done` nghĩa là cả head và hair cap đã export; không đồng nghĩa với mesh đã sẵn sàng để in.

### Local test API — upload ảnh trực tiếp

| Method và route | Xác thực | Tác dụng |
|---|---|---|
| `POST /v1/head/jobs` | Bắt buộc | Nhận multipart gồm `front`, `left`, `right`; `back` tùy chọn. Tạo job local, trả `job_id`, trạng thái và `status_url`. |
| `GET /v1/head/jobs/{job_id}` | Bắt buộc | Xem toàn bộ trạng thái, stages, artifacts, lỗi và thông tin input của job. |
| `GET /v1/head/jobs/{job_id}/logs?stage={stage}` | Bắt buộc | Đọc log hiện tại hoặc log stage cụ thể. Không truyền `stage` để lấy stage hiện tại; log được giới hạn kích thước. |
| `GET /v1/head/jobs/{job_id}/files/{relative}` | Bắt buộc | Tải artifact đã được job công bố, dùng đường dẫn tương đối như `hair/scene.glb`. Không thể dùng route này để đọc tùy ý file nội bộ. |

Cách dễ nhất để gửi ảnh local là đặt đúng tên file trong một folder:

```text
input\04\
  front.png
  left.png
  right.png
  back.png       # tùy chọn
```

Gọi test client; client tự nạp key từ `.env.local`, upload ảnh và với `-Wait` sẽ poll đến khi job kết thúc:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/api/TEST-HEADAPI.ps1 -InputDir ".\input\04" -Wait
```

Có thể bỏ `-Wait` để nhận `job_id` và kiểm tra sau. Từ terminal khác, dùng `job_id` đã trả về:

```powershell
$headers = @{ "X-API-Key" = $env:HEADAPI_API_KEY }
Invoke-RestMethod -Uri "http://127.0.0.1:8090/v1/head/jobs/<job_id>" -Headers $headers
Invoke-WebRequest -Uri "http://127.0.0.1:8090/v1/head/jobs/<job_id>/logs" -Headers $headers
```

Thay `<job_id>` bằng ID thật. Artifact có thể được liệt kê trong response trạng thái; dùng đúng giá trị `path` để tải, ví dụ:

```powershell
Invoke-WebRequest `
  -Uri "http://127.0.0.1:8090/v1/head/jobs/<job_id>/files/hair/scene.glb" `
  -Headers $headers -OutFile ".\scene.glb"
```

## 4. Trạng thái job và lỗi thường gặp

Local test API lưu trạng thái chi tiết như `QUEUED`, `RUNNING`, `MVP_EXPORTED`, `PARTIAL`, `FAILED` hoặc `INTERRUPTED`. Kiểm tra `current_stage`, `stages`, `errors` và artifacts trong status response. `MVP_EXPORTED` chỉ xác nhận pipeline đã xuất đầu và hair cap theo các gate hiện có.

| HTTP status | Ý nghĩa thường gặp |
|---|---|
| `202` | Request hợp lệ, job đã được nhận/đưa vào queue. |
| `401` | Thiếu hoặc sai `X-API-Key`. |
| `404` | Không tìm thấy job, attempt hoặc artifact. |
| `409` | Output chưa sẵn sàng, hoặc dùng lại `job_id` với input/attempt khác. |
| `413` | Request/ảnh vượt giới hạn kích thước. |
| `422` | JSON, UUID, view, URL hoặc ảnh không hợp lệ; thiếu view bắt buộc. |
| `429` | Hàng đợi job đã đầy; thử lại sau khi job đang chạy kết thúc. |
| `503` | Chưa cấu hình API key hoặc prerequisite/pipeline chưa sẵn sàng. |

Các lỗi của một stage cụ thể thường nằm trong status response và file log tương ứng. Dùng endpoint logs hoặc artifacts thay vì suy luận thành công chỉ từ việc server trả `202`.

## 5. Input, output và giới hạn

- Ba ảnh bắt buộc phải được gán đúng `front`, `left`, `right`; ảnh `back` là tùy chọn. Không suy ra góc chụp từ thứ tự ảnh.
- RDF sử dụng các view phía trước/bên mà stage fit hỗ trợ; hair-cap worker cũng nhận ảnh để fit ranh giới tóc. Xem `rdf_views_actually_used` và các report trong artifacts để biết job thực tế đã dùng gì.
- GLB scene hiện xuất head và hair cap thành các component riêng, không Boolean-union. Đây không phải API assembly thành một khối.
- `ready_for_machine` cố ý là `false`. File STL/GLB hoặc trạng thái `done` không phải chứng nhận độ chính xác, khả năng in, an toàn, hay hiệu chuẩn máy.
- Pipeline phụ thuộc weights, parser và các Python environments được khai báo trong `configs/headapi.json`.
- Không commit ảnh cá nhân, weights, output `runs/`, API key hoặc `.env.local`.

## 6. Các workflow khác trong project

Project hiện giữ thêm Pipeline V2.3 độc lập với HeadAPI:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts/experiments/RUN-PIPELINE-V23.ps1
```

Xem [hướng dẫn Pipeline V2.3](./PIPELINE-V23-START-VI.md) và [ghi chú phương pháp](./PIPELINE-V23-METHODS.md). Tài liệu/snapshot lịch sử được lưu dưới [`_archive/project_cleanup_20261030_182137/`](./_archive/project_cleanup_20261030_182137/).

Giữ `runs/` vì đây là dữ liệu kết quả của các lần chạy. Các môi trường còn dùng gồm `.venv`, `.venv-headapi`, `.venv-rdf` và `.venv-triposr`; `.venv-mica` đã được gỡ. `configs/mica.local.json` vẫn còn và có thể tham chiếu tới môi trường đã gỡ.
