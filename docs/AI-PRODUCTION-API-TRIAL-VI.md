# Production API: baseline thử nghiệm cục bộ

## Quyết định kỹ thuật

Dùng MICA để dựng geometry mới từ ảnh front của mỗi job, giữ quy trình phân tích ba ảnh của Production, rồi chuyển màu trực tiếp theo texel với giới hạn bù màu ±12 sRGB. Không dùng lại UV/texture RDF của input04 cho một người mới. Đây là baseline đã kiểm chứng trên các nhóm ảnh hiện có, chưa phải kết luận model tối ưu toàn cục. Chưa fine-tune hay thay weights.

Worker và config mới tách khỏi luồng RDF cũ. API giữ hợp đồng job/attempt của backend, xử lý tuần tự, lưu ảnh gốc, ảnh chuẩn hóa, checksum, config snapshot, model pins và log. Lỗi validation không được xuất thành job thành công. MICA và Blender chạy trong subprocess; API không nạp Torch.

## Kết quả HTTP và GLB

Lượt 01: job `072cc6a1898f45c7a7be769c660a966d`, ảnh thật `input/04`, multipart HTTP 202 → poll `/jobs/{id}` thành `done` → tải GLB thật 38.037.696 byte. Tải sai attempt trả 404. Geometry master, ba texture PBR gốc và roundtrip đã qua kiểm tra của Production. Audit độc lập xác nhận 20.259 tam giác mặt, tọa độ thế giới/UV trong sai số 1e-7 và ảnh atlas giải mã giống nhau giữa master và bản web.

Bản web lượt 01 chỉ giữ 8% lưới thân. Khi render, vùng mũ/tai/viền cằm góc cạnh hơn master. Vì vậy policy web v2 chuyển sang giữ 25% lưới thân; mỗi lượt giữ nguyên thư mục riêng. Báo cáo cuối, ảnh render và kết quả lượt 02 nằm trong `runs/prod-api-01/review.html` và `runs/prod-api-01/attempt-02/`.

Lượt 02: job `87721f2dc2664b6996ffac6aca896554`, HTTP 202 → done → tải thành công 46.320.632 byte (khoảng 44,2 MiB), sai attempt trả 404. Thời gian từ bắt đầu đến hoàn tất khoảng 138 giây trên lượt máy này, không phải benchmark tổng quát. Audit GLB đã chạy trong worker trước khi công bố thành công. Hai lượt dựng mới cùng input tạo cùng SHA-256 master trong lần kiểm chứng này. Ba texture PBR của master giữ nguyên; kiểm chứng không mở rộng thành khẳng định print-ready.

Master đầy đủ: `texture/face_transferred.glb`. Bản web: `render/scene.glb`, phải nhỏ hơn giới hạn 50 MiB của backend. Bản web giảm lưới thân để truyền tải, không dùng làm hồ sơ in. Các texture thân đang chia sẻ với material mặt nên lượt này giữ nguyên; không tuyên bố đã giảm texture thành 1024 nếu report không ghi nhận resize.

Đã chạy HTTP multipart thật và các endpoint status/download đúng hợp đồng backend. Chưa chạy E2E từ giao diện web qua Java, Neon và Cloudinary trong lượt này; chưa kiểm chứng giới hạn upload của tài khoản Cloudinary. Bộ contract tests của API cũng kiểm tra đường JSON `/jobs` với test doubles cho tải ảnh từ xa; đó không phải bằng chứng upload Cloudinary thật.

## Chạy lại

API thử nghiệm dùng cổng 8090. Nếu đang chạy đúng tiến trình này thì không mở thêm instance:

```powershell
.\scripts/api/START-HEADAPI-DEMO.ps1 -ConfigPath configs/headapi-production.json
```

Key đọc từ `.env.local`/`.env` hoặc biến môi trường hiện tại và phải khớp `AI_API_KEY` của backend. Không đưa key vào frontend hoặc URL. Health: `http://127.0.0.1:8090/health`; schema: `/docs`.

Để backend dispatch/sync thật, kiểm tra cấu hình môi trường của backend:

```dotenv
AI_BASE_URL=http://127.0.0.1:8090
AI_DISPATCH_ENABLED=true
AI_SYNC_ENABLED=true
AI_CONNECTION_CHECK_ENABLED=true
AI_EXECUTION_MODE=real
AI_MOCK_ENABLED=false
```

Giữ key riêng trong cấu hình backend. Không commit `.env`. Lượt này không tự sửa cấu hình backend/frontend, quota hay trạng thái thanh toán.

`POST /jobs` nhận UUID job/attempt và ba URL HTTPS front/left/right trên host cho phép. `POST /v1/head/jobs` nhận ba file và optional `attempt_id`, tiện test local. `GET /jobs/{id}` trả trạng thái; `GET /jobs/{id}/outputs/glb?attempt_id=...` trả bản web khi hoàn tất. Endpoint `/validate-image` mà backend mới gọi chưa được bổ sung ở API này; precheck hiện chỉ nhận phản hồi lỗi của adapter và gợi ý, không có bằng chứng kiểm tra ảnh đơn thành công.

## Giới hạn và bước tiếp

Kết quả vẫn `NEEDS_VISUAL_REVIEW`, `ready_for_machine=false`, chưa chứng minh độ giống hay khả năng in. Kính trong mẫu là phụ kiện template; không phải kính tái dựng từ ảnh. Rìa mặt còn chênh màu với đầu mẫu, vùng tai/cằm ngoài patch cần xử lý tiếp. Không dùng texture đẹp để kết luận geometry giống người.

Ưu tiên tiếp theo: blend vùng chuyển tiếp có mask ngữ nghĩa, kiểm chứng nhiều người chưa dùng để chỉnh tham số, rồi kiểm tra upload thật và một lượt web→backend→AI→Cloudinary. Nhánh geometry RDF còn bị guard từ chối phải sửa contract/ánh xạ bề mặt trước; không nới guard để ép thành công.

Các test mới cho raster/provenance, pin/lifecycle và audit GLB đều chạy riêng; contract API 34 tests PASS. Full suite và architecture được chạy lại, log nằm trong `runs/prod-api-01/`. Các lỗi full suite đã có ở nhóm hair manifest thiếu launcher, im2 Linux/pyhocon và mocked workspace resume; không ghi nhận toàn suite PASS.

Kết quả full suite UTF-8: 547 tests, 1 failure, 4 errors, 1 skipped; architecture PASS. 12 tests mới cho texel/Production lifecycle/audit PASS. Log chạy theo encoding Windows mặc định có thêm lỗi decode; dùng `python -X utf8 -m unittest discover -s tests -v` để phân biệt lỗi encoding với các lỗi còn lại.
