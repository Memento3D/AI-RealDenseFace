# Handoff: tích hợp Spring BE với Head Reconstruction AI

Tài liệu này dùng để tiếp tục công việc trong workspace/repository Spring BE. Mục tiêu là sửa BE ít nhất có thể để gọi AI service hiện có, không thay thế kiến trúc reconstruction vừa được cập nhật.

## 1. Yêu cầu cho chat/agent ở repository BE

Trước khi sửa code:

1. Kiểm tra `git status`, branch, commit hiện tại và diff chưa commit. Không reset/revert hoặc ghi đè thay đổi đang có.
2. Đọc source hiện tại, không dùng các DTO/code cũ trong hội thoại này như sự thật hiện hành. Đã đọc repository BE `D:\FPTU\07\SBA301\Memento3D-Backend`, branch `main`, commit `57a7e5652c0eb6166376806592454bd639d5c904`; worktree sạch tại thời điểm kiểm tra. Nếu branch/commit đã đổi, xác minh lại trước khi sửa.
3. Đọc các tài liệu BE hiện có, ưu tiên `docs/08-reconstruction-flow.md`, `docs/03-authentication-security.md`, `docs/02-database.md`, `docs/04-api-reference.md`, `docs/12-backend-guide.md` và tài liệu deployment/setup nếu liên quan.
4. Đọc `FastApiAdapterIntegrationTest` và các test reconstruction mới trước khi kết luận contract hiện hành. Truy vết luồng thật qua controller → application service → dispatcher/`AiGenerationClient` → status synchronizer → finalizer → ModelOutput; registry hiện phục vụ capabilities, chưa điều khiển dispatch. Giữ nguyên quota, tier, preview, authorization, idempotency, events/cache và response public đang được dùng.
5. Đề xuất thay đổi tối thiểu sau khi đọc source. Không sửa API public/FE, migrations hay provider selection nếu chưa chứng minh cần thiết.

## 2. Những điểm đã xác minh trong source BE

- Public submit hiện có hai alias: `POST /api/v1/scans/{scanId}/reconstruction-jobs` và `POST /api/v1/scans/{scanId}/job`. Request/service vẫn chọn **một** `input_image_id`; `CreatePreviewJobRequest` cũng chỉ có một ID.
- `ReconstructionJobServiceImpl` xác minh ảnh thuộc scan, đã upload, chưa bị exclude và có Cloudinary metadata. Snapshot job hiện giữ metadata của một ảnh. `ScanImage`/`UpdateScanImageRequest` chưa có `view_hint`; `sort_order` không phải nhãn góc chụp.
- AI dispatcher hiện dựng một Cloudinary `image/upload` URL từ `cloudinary_public_id`, rồi gửi DTO cũ gồm `job_id`, `attempt_id`, `input_image_id`, `input_url`, `profile`. URL hiện là public delivery URL; source chưa tạo URL ký hạn ngắn.
- AI contract hiện yêu cầu `input_images` với `front`, `left`, `right` bắt buộc và `back` tùy chọn. FE là nơi gán view hint; BE phải kiểm tra và chuyển tiếp, không suy luận từ thứ tự ảnh.
- BE cho phép các profile như `default`, `preview`, `high_quality_detailed`, trong khi AI hiện chỉ xử lý `profile: "default"`. Chưa có policy AI khác nhau cho tier/profile; không gửi profile BE chưa hỗ trợ sang AI như thể có chất lượng khác.
- Capabilities đang quảng bá `minImages=1`, `maxImages=50` cùng guideline Hunyuan3D, không khớp RDF + hair-cap.
- Đã có `ReconstructionProviderRegistry` và FastAPI adapters, nhưng dispatcher, status synchronizer và finalizer vẫn inject trực tiếp `AiGenerationClient`; registry hiện được controller dùng để trả capabilities. Không giả định đổi provider registry sẽ tự đổi luồng thực thi.
- `FastApiAdapterIntegrationTest` dùng Mockito mock client; test không thực hiện HTTP thật. `HttpAiGenerationClient` đã có các endpoint/headers cần thiết cho health, submit, poll và tải GLB, nên không tạo HTTP client mới nếu chưa có lý do.
- Finalizer đã tải GLB, upload Cloudinary, lưu `ModelOutput` cùng SHA-256 rồi mới đánh dấu job/scan hoàn tất. Giữ nguyên idempotency, attempt checks, quota, tier, preview, authorization, events và cache.
- Config BE lấy `AI_BASE_URL`, `AI_API_KEY`, `AI_DISPATCH_ENABLED` và `AI_SYNC_ENABLED`; dispatch/sync mặc định đang tắt. AI local mặc định chạy port `8090`, nên cần cấu hình base URL có thể truy cập từ BE. Giá trị `AI_API_KEY` phải khớp `HEADAPI_API_KEY` phía AI; không ghi giá trị key vào docs/log/Git.

Các phát hiện trên phản ánh source đã đọc ở commit nêu trên. Trước khi sửa BE, kiểm tra lại diff/branch và đọc đầy đủ test/doc mới nhất; không bypass tier/quota hoặc thay đổi public API/migration ngoài phần cần để truyền 3–4 ảnh.

## 3. Mục tiêu tích hợp

- FE tiếp tục dùng API public của BE và không gọi AI service/Cloudinary trực tiếp.
- BE tiếp tục là system of record cho scan, quyền truy cập, job/attempt, quota, trạng thái và model outputs.
- BE gọi AI service nội bộ qua đường client hiện có; không tạo client song song hoặc gọi FastAPI trực tiếp từ controller. Chỉ đưa dispatch qua provider registry nếu chủ động mở scope refactor.
- Giữ flow bất đồng bộ hiện có: submit → poll/status sync → khi AI `done`, finalizer tải GLB → upload Cloudinary → lưu `ModelOutput` → BE mới đánh dấu job hoàn tất.
- Không báo hoàn tất nếu artifact tải/upload/lưu thất bại. Retry finalizer phải an toàn và gắn đúng `job_id` + `attempt_id`.

## 4. AI contract hiện có

Base URL là HTTPS URL của AI service (khi phát triển có thể là URL ngrok HTTPS). AI service hiện cung cấp:

| Method | Path | Dùng để |
|---|---|---|
| `GET` | `/health` | Health check; JSON có `status: "ok"` |
| `POST` | `/jobs` | Tạo/nhận job, trả HTTP `202` |
| `GET` | `/jobs/{job_id}` | Poll trạng thái |
| `GET` | `/jobs/{job_id}/outputs/glb?attempt_id={attempt_id}` | Tải GLB khi job hoàn tất |

Gửi `X-API-Key` ở các request BE → AI ngoại trừ health check. Không gửi key trong body/query, không đưa key vào FE, log hoặc Git.

### Request BE → AI

```json
{
  "job_id": "UUID do BE cấp",
  "attempt_id": "UUID của attempt hiện tại",
  "profile": "default",
  "input_images": [
    {
      "image_id": "UUID ảnh thuộc scan",
      "input_url": "HTTPS URL Cloudinary",
      "view_hint": "front",
      "sha256": "SHA-256 hex, tùy chọn"
    },
    {
      "image_id": "UUID ảnh thuộc scan",
      "input_url": "HTTPS URL Cloudinary",
      "view_hint": "left"
    },
    {
      "image_id": "UUID ảnh thuộc scan",
      "input_url": "HTTPS URL Cloudinary",
      "view_hint": "right"
    }
  ]
}
```

Điều kiện hiện tại:

- Có đúng 3 hoặc 4 phần tử: `front`, `left`, `right` bắt buộc; `back` tùy chọn. `image_id` và `view_hint` phải duy nhất.
- `view_hint` chỉ nhận `front`, `left`, `right`, `back`. AI không tự nhận diện góc.
- Thiếu một view bắt buộc, gửi 1–2 ảnh, trùng hint hoặc profile khác `default` bị từ chối trước khi queue.
- AI xác minh HTTPS, hostname nằm trong `HEADAPI_ALLOWED_IMAGE_HOSTS` (mặc định `res.cloudinary.com`), không theo redirect, tối đa 12 MiB/ảnh; nếu gửi `sha256`, giá trị phải khớp bytes file tải về.
- Payload BE cũ `{job_id, attempt_id, input_image_id, input_url, profile}` không đáp ứng số lượng tối thiểu và bị từ chối `422`; cần chuyển sang `input_images`.
- AI tải ảnh đồng bộ trong lúc nhận request rồi lưu bản chuẩn hóa trong job. URL cần còn hiệu lực đủ lâu cho submit/download đầu vào; không cần tồn tại hết thời gian chạy GPU sau khi AI đã nhận ảnh thành công.
- BE phải kiểm tra đủ 3–4 ảnh thuộc scan và đang hợp lệ (upload complete, chưa xóa, không bị exclude/failed). FE cung cấp `view_hint`; không dùng `sort_order` để suy đoán góc. Có thể truyền nhãn trong request tạo job và lưu cả danh sách trong `inputSnapshot`; giữ `inputImageId` cũ làm front nếu cần tương thích DB/response, tránh migration nếu snapshot đủ.

### Response AI → BE

Submit trả HTTP `202` và JSON có các field tương ứng:

```json
{
  "job_id": "UUID",
  "attempt_id": "UUID",
  "status": "queued",
  "stage": null,
  "progress": null,
  "error_code": null,
  "error_message": null,
  "created_at": "ISO-8601",
  "started_at": null,
  "completed_at": null
}
```

Poll dùng cùng shape; `status` là `queued`, `running`, `done` hoặc `failed`. BE phải xác nhận `job_id` và `attempt_id` trùng attempt đang xử lý trước khi cập nhật trạng thái/finalize. Output chỉ sẵn sàng ở `done`.

`GET /jobs/{job_id}/outputs/glb?attempt_id=...` trả `model/gltf-binary` (`scene.glb`); endpoint kiểm tra attempt và chỉ cho tải khi status AI là hoàn tất.

## 5. Output và giới hạn sản phẩm hiện tại

- GLB gồm node `RDF_Head` và `Hair_Cap` riêng, không Boolean-union. Có manifest `hair/scene.manifest.json` ở phía AI mô tả component.
- GLB dùng tọa độ/đơn vị native RDF, **không phải mm đã hiệu chuẩn**, không phải chứng nhận printability.
- Finalizer BE có thể tiếp tục lưu một `ModelOutput` với format `glb`; không cần đổi schema/FE để chỉ tải và render scene GLB.
- Nếu FE sau này cần bật/tắt/tùy biến component, xem xét lưu/trả manifest có versioned schema. Không thêm thay đổi đó vào scope tối thiểu nếu FE hiện chỉ cần URL GLB.
- API contract bắt buộc 3 view RDF (`front/left/right`), ảnh `back` thứ tư tùy chọn. Điều kiện số lượng không đảm bảo mọi ảnh đều qua được quality/mask/fitting gates; job vẫn có thể `failed`. Không chuyển failure thành success/partial giả.

## 6. Phạm vi BE dự kiến tối thiểu

Sau khi xem source mới, chỉ thay đổi những điểm cần thiết:

1. **Job input request**: đổi request/service để nhận 3 ảnh bắt buộc và 1 ảnh back tùy chọn. Giữ route/response public nếu có thể; lưu danh sách metadata trong `inputSnapshot`, duy trì field `input_image_id`/DB hiện tại nếu cần tương thích.
2. **View hint từ FE**: nhận `image_id + view_hint` cho từng ảnh tại submit; kiểm tra đúng tập `{front,left,right}` cộng tùy chọn `back`, image ownership, uploaded/non-excluded/non-deleted. Không cần thêm AI classifier hoặc suy luận từ `sort_order`.
3. **Dispatcher**: tạo URL cho từng ảnh, gửi `input_images` và `profile: "default"` tới AI. Giữ profile/tier/entitlement hiện có trong BE; chưa gửi profile khác sang AI cho tới khi pipeline có policy tương ứng. Reuse `HttpAiGenerationClient`/adapter phù hợp, không tạo client song song.
4. **Capabilities**: sửa provider capability/guidelines thành min 3, max 4 và đúng RDF + hair-cap. Không quảng bá Hunyuan hoặc 1–50 cho luồng này.
5. **Provider wiring**: dispatcher/poller/finalizer hiện dùng trực tiếp `AiGenerationClient`; giữ đường chạy hiện tại nếu scope là tối thiểu. Chỉ refactor qua registry khi có yêu cầu chọn provider lúc thực thi rõ ràng.
6. **URL ảnh**: source hiện dùng Cloudinary `image/upload` URL public. Giữ HTTPS/allowlist hostname, không gửi credentials Cloudinary cho AI hoặc ghi URL nhạy cảm vào log. Nếu chuyển private/signed delivery, xác minh tương thích rồi mới đổi.
7. **Config/secret**: cấu hình `AI_BASE_URL` trỏ đến AI service, đặt `AI_API_KEY` bằng cùng secret với `HEADAPI_API_KEY`, và bật `AI_DISPATCH_ENABLED`/`AI_SYNC_ENABLED` trong môi trường tích hợp. Không commit key; không dùng giá trị fallback của application config cho production.
8. **Tests**: cover 3/4 ảnh, reject 1/2 và thiếu required view, duplicate IDs/hints, ảnh ownership/status, JSON/header/key, attempt/status mapping, retry/finalizer và capabilities; giữ test quota/preview/provider/mock hiện hành.

Không vội sửa migration, public route, `ModelOutput` schema hoặc controller nếu có thể dùng `inputSnapshot`, DTO nội bộ và provider adapter hiện có. Nếu phát hiện cần migration, giải thích lý do và thiết kế backward-compatible trước.

## 7. Files cần kiểm tra ở source BE hiện tại

### Reconstruction và provider

- `ReconstructionJobController`
- `CreateReconstructionJobRequest`, response DTOs/capabilities/preview DTOs
- `ReconstructionJobService` và implementation
- `ReconstructionJobDispatcher`
- `ReconstructionJobStatusSynchronizer`
- `ReconstructionJobFinalizerImpl`
- `ReconstructionProvider`, `ReconstructionProviderRegistry`
- `HttpAiReconstructionProvider`, `FastApiReconstructionAdapter`, mock provider
- `FastApiAdapterIntegrationTest` và `ReconstructionAndEntitlementTest` (đọc trước khi đổi provider contract; xác định test nào là unit/integration và những policy nào đang được khẳng định)
- `AiGenerationClient` hoặc DTO/client cũ nếu còn tồn tại
- application config (`application*.yml`/`.properties`) và config binding cho `app.ai.*`
- `memeto3D/.env.example` (chỉ xem tên biến/placeholder, không đưa giá trị secret vào chat/Git)
- `docs/08-reconstruction-flow.md` và `docs/03-authentication-security.md` (đọc nội dung, không chỉ dựa vào tên file)
- migrations liên quan `reconstruction_jobs`, input snapshot, `model_outputs`, preview quota

### Scan images và storage

- `Scan`, `ScanImage` và các enum upload status/input mode
- `ScanController`, `ScanService`, implementation
- `ScanImageRepository`, `UpdateScanImageRequest`
- upload/digital asset service, storage port/adapter và `CloudinaryStorageService`
- quality/exclusion flow để xác định điều kiện chọn ảnh chính xác

### Model output và API public

- `ModelOutput`, repository, `ModelOutputController`
- `ModelOutputService`/implementation
- finalizer và bất kỳ conversion/delivery adapter nào được gọi downstream
- docs/API tests/FE contract hiện tại cho scan reconstruction và outputs

## 8. Các kiểm tra cần chạy ở BE

- Tập trung test reconstruction service, provider adapter, dispatcher/status sync, finalizer, scan image selection và output API.
- Chạy test/build/lint theo hướng dẫn của repo BE.
- Kiểm tra JSON thực tế giữa BE và AI bằng test/mock trước; sau đó smoke test qua ngrok với secret test và ảnh test được phép sử dụng.
- Xác nhận không có URL ký, API key hay dữ liệu ảnh bị ghi vào logs.
- Báo rõ thay đổi nào cần FE/config/migration và giữ ngoài scope các phần không bắt buộc.

## 9. Ghi chú an toàn

- Secret AI: AI đọc `HEADAPI_API_KEY`; BE đọc `AI_API_KEY` rồi gửi cùng giá trị qua `X-API-Key`. Không đọc, copy hay in secret `.env.local` trong workspace AI; application config BE hiện có giá trị fallback, không dùng fallback đó cho môi trường triển khai.
- Không dùng giá trị thật trong test fixture, ví dụ hoặc commit.
- URL Cloudinary hiện do BE tạo dưới dạng public `image/upload`; không mô tả nó như signed URL. Chỉ gửi URL qua HTTPS và tránh log. Nếu thay đổi Cloudinary delivery sang private/signed URL, kiểm tra access type/host allowlist trước; không truyền Cloudinary API secret cho AI.
- Chỉ mở ngrok khi cần kiểm thử; không expose debug/review/log endpoints ra internet nếu không có lý do. API AI hiện yêu cầu key cho mọi route trừ health/docs khi khởi chạy bằng cấu hình chuẩn.
