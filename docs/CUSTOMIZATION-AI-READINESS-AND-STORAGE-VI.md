> Cập nhật 2026-11-05: API fit theo job đã được nối và kiểm chứng. Tài liệu trạng thái/hướng dẫn hiện hành: [MVP-ASSET-STORAGE-AND-AI-API-HANDOFF-VI.md](MVP-ASSET-STORAGE-AND-AI-API-HANDOFF-VI.md). Nội dung dưới đây mô tả mốc kiểm tra trước đó.

# Kiểm tra AI và lưu bộ asset custom MVP

Ngày kiểm tra: 2026-11-05. Trạng thái: **base-only chạy được để thử kỹ thuật; chưa chốt AI cho tính năng custom hoàn chỉnh**.

## Kết quả đã xác minh

- Chế độ thử mới: `configs/headapi-customization-base.json`, launcher `START-HEADAPI-CUSTOMIZATION-BASE.ps1`. Cấu hình assembly cũ không đổi, các checksum nguồn vẫn giữ nguyên. Dữ liệu job thử mới dùng `runs/headapi-customization-base`, không trộn với job cũ.
- Bỏ bước `03_hair_cap` khi tạo base; không lắp mesh tóc hoặc mũ. Body/tay và đầu của từng job vẫn được dùng, không thay đầu bằng người mẫu thử nghiệm.
- Thử lại bước assembly từ dữ liệu job có sẵn, không chạy lại inference: `validation/customization/base-pipeline-check-20261105/attempt-01/scene.glb`, 17.241.824 byte. Audit sau Blender đạt: hình học các thành phần được bảo vệ, UV và pixel atlas của đầu giữ nguyên so với master của lần xuất này; body được giảm chi tiết riêng cho web.
- Không nhầm kết quả này với baseline mặt/cổ local đã duyệt. `local_neck_lip_baseline_integrated=false`, `customization_ready=false`, `ready_for_machine=false`.
- Hash file dùng streaming để không đọc toàn bộ GLB lớn vào RAM. Worker tạo thư mục cha trước khi ghi log.
- Sáu test mới đạt; suite 596 test còn bốn lỗi cũ và một skip: thiếu launcher được manifest tham chiếu, hai lỗi test im2 yêu cầu Ubuntu/WSL, thiếu pyhocon. Log: `maintenance/customization-base-full-tests-20261105.log`. Architecture PASS.

## Những phần còn phải hoàn thành trước khi chốt AI

1. Tổng quát hóa xử lý cổ và điều chỉnh môi theo dữ liệu của từng job, với guard, không áp thông số một người cho mọi ảnh.
2. Chốt model nền/preset sạch. Mũ gốc và tóc hiện được review với hai body/tay khác nhau, chưa coi là phụ kiện bật/tắt tùy ý trên cùng body.
3. Chuẩn hóa canonical asset/anchors và kết quả fit theo đầu/preset. File `fitted-reference.glb` chỉ là ví dụ đã fit của job hiện tại.
4. Có scene manifest, danh sách tương thích, endpoint trả dữ liệu custom và tác vụ export tái dựng từ snapshot có version/hash. Hiện chưa triển khai các endpoint custom này.
5. Kiểm tra thêm đầu vào thực, hình học va chạm, độ dày và quy trình sản xuất; file GLB xuất thành công không đồng nghĩa in được.

Không thay cấu hình API đang dùng sang chế độ thử như thể tính năng custom đã hoàn thiện. Không sửa `published` hoặc `print_ready` để vượt các điểm chặn.

## Vị trí local

- Asset: `assets/custom-library/mvp-review-20261105/`.
- Ảnh có khuôn mặt user: `validation/customization/mvp-review-20261105/`, đã được gitignore; không upload lên thư viện chung.
- Bốn mẫu: graduation-cap, hair-side-part, glasses-round, glasses-alternative. Trạng thái tất cả: local review, chưa phát hành.
- `candidate-thumbnail.png` là thumbnail của nguồn ứng viên, chưa bảo đảm khớp màu/hình học của fitted-reference. Không dùng làm thumbnail chính thức.

## Tự kiểm tra

Chạy từ thư mục gốc AI:

```powershell
.\START-HEADAPI-CUSTOMIZATION-BASE.ps1 -CheckOnly
.\.venv\Scripts\python.exe -B -X utf8 -m tools.customization.audit_library --library assets/custom-library/mvp-review-20261105
```

Audit kiểm tra inventory, checksum, đường dẫn an toàn và node accessory không chứa head/body. Nó không tự duyệt hình ảnh hoặc chứng nhận khả năng in. `integrity=PASS` chỉ nói dữ liệu nguyên vẹn. Chạy thêm `--require-publishable` hiện phải trả exit code 2 vì bộ này chưa đủ điều kiện phát hành.

## Lưu B2 ngay bây giờ: bản nháp private

Mục đích hiện tại là lưu giữ bản review, không phải phát hành. Không cần chờ BE có catalog để backup file, nhưng chưa có record được duyệt thì FE không được công bố.

1. Vào B2 > Browse Files, chọn bucket **memento3d-models-tiendon**. Giữ private.
2. Tạo prefix/thư mục `library-drafts/mvp-review-20261105/`.
3. Upload nội dung thư mục asset local vào prefix này, **giữ đúng các thư mục con**. Nếu web console không giữ cấu trúc khi chọn cả thư mục, tạo từng thư mục bằng New Folder rồi upload file ở đúng cấp; không kéo mọi GLB vào chung một cấp.
4. Cấu trúc cần thấy:

```text
library-drafts/mvp-review-20261105/
  catalog.json
  inventory.json
  fit-profiles-reference.json
  README-VI.md
  library/
    graduation-cap/review-v1/...
    hair-side-part/review-v1/...
    glasses-round/review-v1/...
    glasses-alternative/review-v1/...
```

5. So sánh số file và dung lượng theo inventory. Khi có importer BE, tải kiểm tra checksum rồi mới đăng ký version. Đường dẫn lưu bền là bucket + object key; không dùng URL ký có thời hạn làm định danh.
6. Không ghi đè lần review đã lưu. Khi sửa nội dung, dùng review-v2 hoặc một prefix review mới và tạo inventory mới.

Chưa upload: thư mục runs, input, ảnh private-review, secret, config.snapshot của job. Không dùng prefix `library/releases` cho bản này.

## Cloudinary và Neon ở giai đoạn này

**Cloudinary:** đợi có thumbnail đúng asset đã chuẩn hóa, cùng ánh sáng/góc nhìn rồi mới upload vào `memento3d/library/<asset-id>/<version>/`. Lưu public_id và version do Cloudinary trả về; folder hiển thị không luôn đồng nghĩa với public_id. Không dùng ảnh có khuôn mặt thử nghiệm làm thumbnail thư viện chung.

**Neon PostgreSQL:** chưa nhập tay catalog vào bảng nghiệp vụ cũ. Khi làm BE sẽ migration/import: asset/version, B2 key, thumbnail reference, checksum, trạng thái duyệt, compatibility/fit profile. Custom của user lưu JSON có version; snapshot đơn hàng riêng và bất biến. Không lưu binary GLB vào DB.

Khi đủ điều kiện phát hành, tạo release mới từ file đã duyệt và đăng ký trong BE. Việc lưu draft trên B2 không tự thay đổi trạng thái duyệt hoặc dữ liệu đang được user/order tham chiếu.

## Tài liệu dịch vụ

- B2 web console: https://www.backblaze.com/docs/en/cloud-storage-upload-and-manage-files
- Cloudinary upload: https://cloudinary.com/documentation/dam_upload_store_assets
- Cloudinary folder modes: https://cloudinary.com/documentation/folder_modes
