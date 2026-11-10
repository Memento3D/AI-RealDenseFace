# Gom manifest và dọn folder — 10/11/2026

## Đã thực hiện

- Chuyển 7 JSON manifest ở gốc sang configs/manifests. Không xóa vì runtime nghiên cứu vẫn dùng để kiểm tra source/code.
- Chuyển 6 JSON release/summary từ verification sang maintenance/verification; các report validation đã tracked được giữ tại maintenance/verification/validation.
- Chuyển 26 file tài liệu/evidence còn được manifest tham chiếu khỏi _archive/verification sang docs/legacy hoặc maintenance/verification trước khi dọn.
- Cập nhật đường dẫn nơi đọc manifest, producer path trong cache và test đường dẫn tương ứng. Thuật toán, HTTP contract và các cấu hình pipeline/asset pin giữ nguyên.
- Ghi old/new cùng hash trong manifest-relocations-20261110.json và legacy-evidence-relocations-20261110.json. Hash manifest lồng nhau được cập nhật theo dependency tới khi ổn định; không refresh các entry cũ khác để che drift có sẵn.
- Viết lại README theo luồng assembly hiện tại, setup từng môi trường và prerequisites asset. Bản tham khảo request API trước đó giữ tại docs/guides/API-REQUEST-REFERENCE-VI.md có chú thích phạm vi.

## Đã xóa

9 thư mục: _archive, verification, validation, hai maintenance/runs-archive-* và bốn thư mục tên lỗi Unicode do output trước đây. Đường dẫn tuyệt đối được kiểm tra nằm trong workspace, không có reparse point trong cây xóa. Bốn thư mục tên lỗi không có file, chỉ còn thư mục con rỗng. Xóa thêm một mica.local.before-fix cũ, giữ mica.local hiện hành.

501 file cũ bị loại, tổng 792.796.135 byte. Trước khi xóa hai ZIP, giữ 1.255 mapping của report/config nhỏ và numeric parameters vào objects theo SHA-256 (~112,9 MB). Giữ thêm 121 file parameter/source từ validation (~13,8 MB). Dung lượng thu hồi ròng khoảng 635 MiB, chưa tính các file evidence nhỏ đã chuyển vị trí. Thông số local được ignore Git.

Inventory xóa: maintenance/local-cleanup/20261110/deleted-folders.json. Index thông số: archive-parameters/index.json và validation-parameters/index.json dưới cùng thư mục local-cleanup. Không tạo thêm ZIP backup lớn.

## Giữ lại

Các môi trường, assets, input, vendor, runtime production trong runs và hair-bench. Hair-bench còn là workspace mặc định của launcher nghiên cứu RUN-CURRENT, không coi là folder rác. schemas/tests/viewer là source hoặc contract, không phải output dư.

Các log/evidence legacy được manifest kiểm tra được chuyển thay vì xóa. Chúng không làm assembly hiện tại đổi logic; nhưng việc chạy lại nhánh legacy có thể cần bộ evidence local đó như trước. Không tuyên bố clone sạch kèm đủ weights/asset.

## Kiểm tra

Assembly -CheckOnly và architecture check đạt sau khi gom/xóa. Full suite sau thay đổi ghi tại maintenance/local-cleanup/20261110/unittest-json-cleanup.log. Cần đọc số failure/error thực tế; chưa dùng preflight để tuyên bố inference hay khả năng in thành công.

Kết quả full suite: 602 tests, 2 failures, 4 errors, 1 skipped. So với lượt trước có thêm failure cache revision HairStudio; chạy riêng 11 test HairStudio vẫn còn 1 failure. Trong đợt này, phần implementation HairStudio chỉ đổi thông báo tên launcher ở lượt trước, không đổi thuật toán revision. Không sửa logic cache hoặc resume trong phạm vi gom file. Bốn error còn lại gồm manifest hair thiếu file và nhánh dependency im2mesh. Các log được giữ trong local-cleanup, không che failure hoặc tạo file giả để đạt test.
