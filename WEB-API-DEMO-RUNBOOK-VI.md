# Chạy kiểm chứng web → BE → Head API cho demo

## Quyết định sau khi đối chiếu

- Input04: giữ geometry/camera baseline; các candidate gần đây còn regression vùng môi profile.
- Capture02: camera-only có cải thiện detector residual, nhưng chưa đủ để bật refinement tự động cho mọi bộ ảnh.
- Bản demo mới giữ **RDF geometry của từng job mới**, không chọn sẵn mesh theo tên người/bộ ảnh. Bật generic UV texture + strict visibility + confidence/provenance và ghép vào scene head/hair hiện có. Không mount precomputed experimental model làm kết quả inference.
- Trọng số/model pins không được đổi. Không fine-tune, không làm thêm body tốt nghiệp hay đổi cách fit tóc trong đợt này.

## Những gì đã nối

Chọn `configs/headapi-demo.json`, `face_render_policy=strict_uv_v1`. Jobs mới ở `runs/headapi-demo`, queue tối đa một job. Luồng:

Đợt chuẩn bị trình bày mới bật thêm `presentation_policy=harmonized_head_v1`: màu tóc cho cap hiện có, nền scalp và màu da ước lượng dưới cằm/cổ. Lớp này chỉ phục vụ appearance demo; giữ geometry và kính, lưu nhãn/confidence inferred riêng. Đặt `presentation_policy=none` rồi restart để quay về UV quan sát cho nghiên cứu.

1. `01_rdf_observations` — crop/landmarks thật.
2. `02_rdf_fit` — RDF hiện tại.
3. `02b_rdf_export` — giữ export cũ để đối chiếu.
4. `03_hair_cap` — luồng tóc cũ, không đổi geometry.
5. `03b_uv_texture` — strict continuous visibility 1e−6 native units; per-job conservative RGB gains 0,85–1,18; UV atlas, confidence và conservative finishing.
6. `04_scene_export` — ghép UV head với chính cap cùng RDF bundle; xác nhận hai geometry không đổi, có embedded base-color texture, không external URI.

Progress BE đi theo preprocessing → shape_generation → texturing → exporting, không quay ngược stage khi fit tóc. Endpoint `/jobs/{job_id}/outputs/glb?attempt_id=...` trả `render/scene.glb` ở policy mới. Contract JSON/headers/attempt giữ nguyên.

UV hoặc scene fail sẽ không báo done, không âm thầm trả scene màu vertex cũ. Output whitelist được publish cùng terminal status để BE tải ngay khi thấy done. Cấu hình `configs/headapi.json` cũ vẫn dùng legacy rendering; muốn rollback thì chạy lại bằng config đó với job mới.

## Khởi động AI local

Từ thư mục repo này, dùng PowerShell:

```powershell
# Điền HEADAPI_API_KEY trong .env.local giống app.ai.api-key bên BE.
# Launcher tự đọc file; không in giá trị khóa.
.\START-HEADAPI-DEMO.ps1
```

Thứ tự ưu tiên: biến môi trường terminal → `.env.local` → `.env`. Chỉ nạp `HEADAPI_API_KEY` và `HEADAPI_ALLOWED_IMAGE_HOSTS`; không thực thi nội dung file. `.env.example` là mẫu, không được tự nạp. Có thể kiểm tra riêng mà chưa chạy server:

```powershell
.\START-HEADAPI-DEMO.ps1 -CheckOnly
```

Khóa không được để trống và phải giống BE. Không cần gửi khóa vào chat hoặc đặt vào frontend.

Launcher chạy foreground ở `http://127.0.0.1:8090`, không tự mở hoặc dừng process khác. Chỉ chạy một Head API phục vụ GPU; nếu port đang được dùng, kiểm tra phiên server cũ trước. Không sửa source trong khi job đang chạy vì code hash được kiểm tra theo attempt.

Ở terminal khác:

```powershell
Invoke-RestMethod http://127.0.0.1:8090/health
```

Kỳ vọng `status=ok`, `backend_ready=true`, `face_render_policy=strict_uv_v1`, `presentation_policy=harmonized_head_v1`. Phải restart server sau khi đổi config. Health kiểm tra prerequisites, không chứng nhận chất lượng ảnh/GPU/printability.

Có thể smoke test AI trực tiếp bằng client hiện có, nhưng thao tác này sẽ chạy reconstruction thật và tạo job mới:

Nếu dùng client ở terminal khác, terminal đó cũng cần `HEADAPI_API_KEY` cùng giá trị; biến môi trường của cửa sổ server không tự xuất hiện trong cửa sổ khác.

```powershell
.venv\Scripts\python.exe -m tools.headapi.client --input input/04 --url http://127.0.0.1:8090 --wait
```

Chưa chạy lệnh này trong đợt chuẩn bị; ưu tiên giữ lượt kiểm chứng mới cho luồng web đầy đủ.

## Cấu hình BE cần đối chiếu khi chạy web

Đã đọc source BE hiện tại; không sửa file BE/web trong đợt này.

- `app.ai.base-url`: URL BE thực sự truy cập được tới Head API. `http://127.0.0.1:8090` chỉ đúng nếu BE cùng host. Nếu BE chạy container/máy khác, localhost là chính nơi đó, phải cấu hình địa chỉ phù hợp.
- `app.ai.api-key`: cùng khóa với `HEADAPI_API_KEY`.
- `app.ai.execution-mode=real` cho lượt kiểm chứng inference thật. Provider `HTTP_AI` hoặc `FASTAPI` hiện cùng gọi HTTP client thật; không dùng `MOCK_AI` để xác nhận pipeline.
- AI input URLs phải HTTPS trên host cho phép; mặc định `res.cloudinary.com`. Nếu dùng custom host, cần cấu hình allowlist tương ứng thay vì tắt kiểm tra.
- Dispatcher hiện gửi `profile=default` sang AI. Web gửi preview profile cho BE không yêu cầu mở thêm profile preview ở AI.

Backend `HttpAiGenerationClient` tải GLB theo đúng job_id + attempt_id, giới hạn 50 MiB. Scene rehearsal mới khoảng 2,39 và 2,17 MiB, texture nhúng trong GLB.

## Các bước kiểm chứng từ web

1. Dùng trang tạo keepsake và **scan mới**, upload ảnh khác nhau cho front/left/right; back tùy chọn. FE đang gửi upload kèm `view_hint`; consent/quota do luồng web/BE quản lý.
2. Xác nhận preview request `/reconstructions/previews` trả 202/job ID. Không dùng trang `/reconstruction/demo` hoặc output mẫu để chứng minh inference mới.
3. Theo dõi BE dispatch sang AI: `POST /jobs` với `job_id`, `attempt_id`, `profile=default`, ba hoặc bốn `input_images`, header `X-API-Key`.
4. Trong `runs/headapi-demo/<job_id>`, xem `config.snapshot.json`, `job.json` và log từng stage. Phải thấy `03b_uv_texture` và `04_scene_export` thực sự chạy, không chỉ RDF/hair cũ.
5. Khi AI trả done, BE tải GLB theo attempt rồi lưu model output/delivery URL. Xác nhận job/attempt này tương ứng scan mới; không tái sử dụng URL/model cũ như bằng chứng inference mới.
6. Mở model trên web, kiểm tra texture mặt và node tóc, xoay hai profile. `Keepsake3DViewer` hiện clone material thay vì bỏ texture map; chưa kiểm chứng WebGL/browser trong đợt này.
7. Đối chiếu file web tải với `render/report.json`/`scene_sha256` qua BE delivery nếu file không bị biến đổi. Giữ screenshots + job/attempt + stage logs cho demo.

Một điểm cần nhìn khi test browser: viewer đang clamp camera distance tối thiểu 2,5 trong khi head native height khoảng 0,33–0,36. Mô hình có thể hiện nhỏ; controls cho zoom xuống 0,8. Chưa thay viewer hoặc scale geometry để tránh nhầm với scale in vật lý. Nếu cần chỉnh camera fit, làm ở bước kiểm chứng web tiếp theo.

## Phân biệt các mức kiểm chứng hiện có

- **Đã làm:** 10 test mới cho orchestration, failure handling, đúng output/attempt, embedded UV material, geometry binding, memory guards và publish artifact trước done.
- **Đã làm thật:** chạy generic UV/scene workers trên hai bộ RDF + hair artifacts thật đã có; xác nhận head/cap không đổi, texture nhúng và nguồn giữ nguyên checksum.
- **Đã làm:** in-process API output contract replay bằng read-only fixture service và scene thật; đúng bytes, đúng attempt, thiếu key 401, sai attempt 404. Không phải test submit/reconstruction qua mạng.
- **Chưa làm:** lượt inference mới từ ảnh upload trên web, Spring dispatch/poll/finalize thật, Cloudinary fetch/upload mới và browser rendering.

Kết quả rehearsal: `runs/experiments/api-demo-uv-01/review.html`, `report.json`, `http-contract.json`. Không dùng rehearsal như bằng chứng web E2E đã pass.

## Lỗi cần phân biệt khi rehearsal web

- 401: so lại khóa BE và AI, không đưa khóa vào FE.
- 422: xem input_images/view_hint, ảnh trùng, URL hoặc hash; giữ đầy đủ front/left/right.
- 429: queue demo đang có job; đợi hoàn tất rồi submit scan mới.
- `03b_uv_texture` fail: đọc log UV/pins/RAM; không chuyển scene cũ thành done.
- AI done nhưng BE không có model output: kiểm tra download theo attempt và bước storage/finalizer của BE.
- GLB local có texture nhưng web không thấy: kiểm tra file URL thực, cache/delivery, material và camera viewer; không kết luận lỗi reconstruction trước.

Mỗi attempt giữ nguyên artifacts; retry cho demo dùng scan/job mới. Không xóa các attempt nghiên cứu hoặc dùng scene demo cache thay cho lượt infer thật.

## Giới hạn demo

`done` nghĩa là đã xuất render scene, không có nghĩa giống đúng người hoặc in được. Scene gồm head và cap riêng, không boolean union/manufacturing mesh; `ready_for_machine=false`. Vùng da không quan sát lớn vẫn xám, scale native chưa hiệu chuẩn mm. Đợt này chuẩn bị luồng render và contract cho web, chưa tự nhận production-ready.
