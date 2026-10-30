# Dọn cấu trúc project — giữ mốc V2.3

Áp dụng cho ZIP source-review ngày 28/10/2026 bạn đã gửi.
Đây là gói bổ sung công cụ dọn, không phải source thay thế hay pipeline mới.

## Cách áp dụng

1. Giải nén nội dung ZIP trực tiếp vào:
   `D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`
   để `CLEAN-PROJECT.ps1` nằm cạnh `RUN-PIPELINE-V23.ps1` và `.venv`.
2. Đóng những tiến trình thử nghiệm đang chạy. Xem trước nếu cần:
   `powershell -ExecutionPolicy Bypass -File .\CLEAN-PROJECT.ps1`
3. Dọn theo đúng danh sách đã kiểm tra:
   `powershell -ExecutionPolicy Bypass -File .\CLEAN-PROJECT.ps1 -Apply`
4. Xem `PROJECT-MAP-VI.md` để biết vị trí code hiện hành.

Công cụ chuyển 57 tài liệu/launcher lịch sử từ gốc sang
`_archive/source_cleanup_<timestamp>/docs` hoặc `launchers`.
Không xóa file; không đổi Python, config, checksum V2.3, ảnh, kết quả, model,
venv hay cache OCR. Mọi file cần chuyển phải khớp SHA256 trong danh sách.
Nếu file đã thay đổi hoặc thiếu, công cụ dừng trước khi di chuyển.
Nếu thao tác di chuyển gặp lỗi, công cụ cố hoàn tác từ journal; mất điện/kill
đột ngột cần chạy lệnh Rollback bên dưới. Không bảo đảm atomic cả 57 file.
Các launcher lịch sử đã chuyển không chạy được trực tiếp ở vị trí archive:
chúng dùng PSScriptRoot và chưa được viết lại. Hoàn tác trước khi chạy chúng.

## Hoàn tác

Thay đường dẫn bằng ARCHIVED được in khi dọn:
`powershell -ExecutionPolicy Bypass -File .\CLEAN-PROJECT.ps1 -Rollback ".\_archive\source_cleanup_<timestamp>"`

Hoàn tác từ chối ghi đè nếu file gốc/archived đã bị thay nội dung.
Trường hợp bản gốc giống hệt đã tồn tại, giữ nguyên cả hai bản; không xóa.
Không dùng git clean/reset để dọn: nhiều code hiện tại chưa được Git theo dõi.

## Chạy model về sau

`powershell -ExecutionPolicy Bypass -File .\RUN-CURRENT.ps1`

Lệnh này gọi launcher V2.3 nguyên bản với workspace đúng ở
`hair-bench\input04\head.workspace.html`. Không cần chạy lại chỉ để dọn source.
Nếu đổi dữ liệu, truyền `-Workspace "duong-dan-moi"` rõ ràng.

## Vì sao giữ src/scripts?

Source chỉ vài MB. Manifest V2.3 cần module từ nhiều phiên bản trước.
Một số launcher vẫn được test đọc ở vị trí gốc nên giữ nguyên. Việc xóa hoặc
đổi chỗ module sẽ là refactor riêng, cần đầy đủ assets/vendor/viewer và SDK.
Gói review cố ý không có chúng; không tạo file giả để làm test xanh.

## Kiểm tra đã thực hiện

- 63/63 file runtime khớp manifest V2.3 của ZIP đầu vào.
- Xem trước, áp dụng, hoàn tác trả về byte gốc và từ chối file bị thay: đạt.
- scripts/check_architecture.py: PASS trước và sau.
- unittest discover -s tests -v: 291 tests; trước và sau cùng 4 failures,
  38 errors, 1 skipped (248 pass). Các tên test lỗi không thay đổi.
- Các lỗi trong gói review gồm thiếu cv2, pyhocon, assets/hair/library.json,
  viewer/hair-on-head/index.html. Xem log đi kèm; không tuyên bố toàn bộ test pass.
- Thử trên bản sao Linux. Chưa thực thi PowerShell/NTFS trên Windows của bạn;
  không tuyên bố đo tốc độ Windows hay kiểm chứng in vật lý.

## Dung lượng ổ D

Di chuyển trong cùng ổ không giải phóng GB. `pip cache purge` xử lý riêng khoảng
8.34 GiB cache pip đo được. Gói dọn này không xóa Hugging Face, Paddle OCR,
Hunyuan, .venv-rdf hay .venv-mica; cần quyết định theo nhu cầu project.
