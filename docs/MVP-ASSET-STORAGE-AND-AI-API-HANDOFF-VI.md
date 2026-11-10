# Lưu bộ asset MVP và bàn giao API AI cho BE

Ngày: 2026-11-05. Kết luận: **đủ điều kiện chuyển sang BE cho MVP preview/custom**, chưa phải toàn bộ luồng đặt in hoàn chỉnh.

## Phạm vi đã kiểm chứng

- Mẫu nhân vật: graduation-standing-base-v1.
- Ba asset: graduation-cap@v1, hair-side-part@v1, glasses-round@v1. Mũ/tóc loại trừ nhau; kính có lựa chọn null.
- Đầu/mặt/cổ giữ baseline người dùng đã chấp nhận. Lỗi bề mặt mũ và hairline đã ghi trong catalog, không dùng làm lý do mở lại tối ưu hình ảnh trong giai đoạn này.
- AI tạo base rồi tự fit theo từng job, không dùng fitted GLB của một người cố định cho tất cả ảnh.
- Mũ: geometry_mode=shared_transform, mesh dùng chung + matrix theo đầu. Tóc/kính: geometry_mode=fitted_geometry, file nhỏ riêng vì tóc thay kích thước và càng kính có biến dạng cục bộ.
- Lượt kiểm chứng mới đi qua API multipart thật trong FastAPI TestClient, tái chạy inference từ ba ảnh: job ead03dd2a83a4317922f2e7b1f6e85f4, attempt 1a232fdd-e71f-4a0c-bf69-dd550d48caf3, khoảng 102,5 giây. Sáu stage đạt, không dùng output trung gian cũ.
- Test catalog, base, manifest, fitted asset, shared asset, thumbnail đều đúng checksum. Sai quyền/sai attempt/asset lạ bị từ chối. Ghép đúng file và matrix API trả về rồi render mũ+kính, tóc+kính để đối chiếu.
- Report/review: validation/customization/api-mvp-check-20261105/ead03dd2a83a4317922f2e7b1f6e85f4/index.html.
- 21 test custom đạt. Full suite 602 test còn bốn lỗi môi trường cũ và một skip; architecture PASS. Log maintenance/customization-api-full-tests-20261105.log.

Chưa kiểm chứng nhiều người khác nhau. Fit vượt giới hạn/không đủ clearance sẽ không công bố custom thành công. Chưa kiểm thử xuyên Spring/ngrok/B2/FE. Chưa có endpoint xuất snapshot đặt in; phần này thuộc giai đoạn đơn hàng.

## 1. B2: lưu toàn bộ bộ asset dùng chung

Thư mục nguồn:

```text
assets/custom-library/mvp-selected-preview-v1/
```

Bucket: memento3d-models-tiendon, private. Endpoint: https://s3.us-east-005.backblazeb2.com, region us-east-005.

Vào B2 > Browse Files > chọn bucket. Tạo prefix **library/mvp-v1/** bằng New Folder, sau đó upload nội dung thư mục nguồn và giữ nguyên cấu trúc:

```text
library/mvp-v1/
  catalog.json
  fitting-profile.json
  inventory.json
  README-VI.md
  assets/
    graduation-cap/v1/
      preview.glb
      thumbnail.png
      manifest.json
    hair-side-part/v1/
      preview.glb
      thumbnail.png
      manifest.json
      license-evidence.html
    glasses-round/v1/
      preview.glb
      thumbnail.png
      manifest.json
      license-evidence.html
```

Có 15 file. Nếu chọn thư mục mà web console không giữ cấu trúc, tạo từng thư mục rồi upload đúng cấp; không dồn ba file preview.glb cùng tên vào một thư mục. Không upload một ZIP thay cho các object riêng mà ứng dụng cần đọc.

B2 giữ bản gốc thumbnail như một phần của bộ asset để bảo toàn inventory; bản thumbnail tối ưu phục vụ UI sẽ qua Cloudinary. Phần lặp dung lượng ảnh là nhỏ và có chủ đích.

Không upload thư mục validation, runs, inputs hoặc các scene có mặt user vào prefix thư viện này. Base và file fit của mỗi user sẽ được BE lưu riêng sau khi nối API.

### Kiểm tra sau upload

- Kiểm tra đủ 15 file và đúng đường dẫn. Bản kế hoạch chính xác nằm ở maintenance/mvp-asset-upload-plan-20261105.json, có object key, byte size và SHA256 của từng file.
- Tải lại các GLB từ B2, chạy `Get-FileHash -Algorithm SHA256 '<duong-dan-file-da-tai>'` và so với plan/catalog. Có thể tải cả bộ về để kiểm tra tất cả checksum.
- Không đổi bucket sang public để lấy link. BE lưu bucket+object key và cấp URL tải có thời hạn sau khi kiểm tra quyền.
- Không ghi đè v1 khi sửa nội dung. Dùng v2 và tạo catalog/pin mới. `published=false` hiện là chưa đăng ký/công bố trong BE, không ngăn lưu private trên B2.

## 2. Cloudinary: ba thumbnail cho UI

Upload đúng ba file sau trong Media Library:

```text
assets/graduation-cap/v1/thumbnail.png
assets/hair-side-part/v1/thumbnail.png
assets/glasses-round/v1/thumbnail.png
```

Gợi ý nhóm thư mục:

```text
memento3d/library/graduation-cap/v1/
memento3d/library/hair-side-part/v1/
memento3d/library/glasses-round/v1/
```

Sau mỗi lần upload, giữ lại **public_id, asset_id, version, secure_url** thực tế. Folder hiển thị không luôn đồng nghĩa với public_id, đặc biệt khi dùng dynamic folders. Không cần gửi secret; chỉ cần metadata/URL ảnh để đăng ký catalog BE.

FE/app sau này dùng ảnh nhỏ khoảng 256 px và định dạng/chất lượng tối ưu qua Cloudinary. Không đưa GLB lên Cloudinary. Không dùng ảnh review có mặt người làm thumbnail chung.

## 3. Neon PostgreSQL: chờ migration/import của BE

Không tạo bảng hay nhập record rời rạc bằng SQL thủ công trong bước lưu file. BE sẽ quản lý:

- Catalog, asset/version, B2 object key, Cloudinary public_id/version, checksum và trạng thái duyệt.
- FitResultVersion theo base/job/attempt/catalog/profile, gồm source hash và fitted hash/matrix.
- Một draft JSON đang chỉnh cho model của user; Lưu cập nhật draft, có optimistic revision.
- Snapshot bất biến gắn order item khi xác nhận thiết kế, tách khỏi draft. Không tạo ModelVersion hoặc GLB sau mỗi click.

## 4. Chạy AI để BE kết nối

Từ thư mục gốc Memento3D-Head-Reconstruction:

```powershell
.\scripts/api/START-HEADAPI-CUSTOMIZATION-BASE.ps1 -CheckOnly
.\scripts/api/START-HEADAPI-CUSTOMIZATION-BASE.ps1 -Port 8090
```

Launcher dùng HEADAPI_API_KEY từ environment/.env.local/.env, không cần gửi key trong chat. Không chạy hai API cùng port. Job của chế độ này ở runs/headapi-customization-base, nên job thuộc thư mục assembly cũ không tự xuất hiện trong API mới.

Terminal khác:

```powershell
ngrok http 8090
```

Giữ URL HTTPS ngrok để cập nhật AI_BASE_URL của BE ở giai đoạn nối BE. HEADAPI_API_KEY phải khớp app.ai.api-key phía BE. Không đưa key này vào FE/app.

Health không cần key:

```powershell
Invoke-RestMethod http://127.0.0.1:8090/health
```

Kỳ vọng backend_ready=true, assembly_output_mode=customization_base, customization_fit_enabled=true. Health đạt không có nghĩa một job đã xong hoặc model in được.

## 5. Hợp đồng API đã có để BE dùng

Tất cả endpoint dưới đây trừ health đều cần X-API-Key:

| Endpoint | Mục đích |
|---|---|
| POST /jobs | Contract BE hiện có: job/attempt và URL ảnh Cloudinary |
| POST /v1/head/jobs | Multipart ảnh, dùng kiểm thử local |
| GET /jobs/{jobId} | Trạng thái, output_kind và customization_ready |
| GET /jobs/{jobId}/outputs/glb?attempt_id=... | Model nền đầu+body |
| GET /customization/catalog | Danh sách version asset cho BE import |
| GET /customization/assets/{assetId}/preview | GLB dùng chung |
| GET /customization/assets/{assetId}/thumbnail | Thumbnail gốc |
| GET /jobs/{jobId}/customization?attempt_id=... | Scene manifest sau fit |
| GET /jobs/{jobId}/customization/assets/{assetId}?attempt_id=... | Geometry đúng theo manifest; mũ dùng chung, tóc/kính fit riêng |

Khi job done và customization_ready=true, BE tải base và manifest. Chỉ tải/lưu geometry riêng cho entry fitted_geometry; shared_transform tham chiếu asset library cùng SHA. Matrix là column-major trong hệ glTF Y-up, áp dưới cùng ModelRoot. Không tự căn giữa từng phụ kiện.

Thông tin source/manifest/checksum là dữ liệu có version. BE không lấy fitted file của job cũ cho một job khác, không dùng URL ký tạm thời làm định danh lâu dài và không cho FE gọi Python bằng server key.

## 6. Thứ tự làm BE sau khi lưu xong

1. Migration/catalog importer và kiểm tra object tồn tại/checksum; ánh xạ thumbnail Cloudinary.
2. Mở rộng đồng bộ AI: nhận base + fit manifest + fitted assets, lưu B2 và FitResultVersion đúng job/attempt.
3. API catalog/scene cho FE, lưu và mở lại draft có revision, kiểm tra tương thích và quyền sở hữu.
4. Gắn snapshot với order item, export khi xác nhận, duyệt model và liên kết đúng artifact cho print job. Chưa dùng preview 1024 px hoặc file scan hở như master in tự động.
5. Test xuyên BE/ngrok/B2/FE trước khi coi toàn bộ tính năng MVP hoàn tất.

Tài liệu chính thức: B2 https://www.backblaze.com/docs/en/cloud-storage-upload-and-manage-files ; Cloudinary https://cloudinary.com/documentation/upload_parameters .
