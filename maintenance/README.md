# Maintenance records

- `verification/`: JSON/evidence lịch sử được giữ khi gom cấu trúc; một số file được manifest legacy tham chiếu.
- Các file báo cáo cleanup/audit ở mức này ghi lại quyết định và provenance.
- `local-cleanup/`: inventory, thông số và log cục bộ, được ignore Git. Các bản dữ liệu người dùng/numeric checkpoint đã giữ từ ZIP cũ nằm tại `local-cleanup/20261110/archive-parameters/`, tra bằng `index.json` và SHA-256.
- Các report/parameter của validation cũ nằm tại `local-cleanup/20261110/validation-parameters/`.

Script vận hành nằm trong `scripts/maintenance/`. Không commit file ZIP backup, environments hoặc bản sao output để lưu lịch sử; kiểm tra `.gitignore` và dung lượng staged trước khi push.
