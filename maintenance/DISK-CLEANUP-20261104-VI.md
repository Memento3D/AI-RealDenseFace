# Dọn cache trước thử tóc — 2026-11-04

Đã tạm dừng tải/thử tóc theo yêu cầu. Phạm vi dọn: workspace Head Reconstruction, chưa dọn toàn ổ D hay repo khác.

- Xóa `runs/prod-win-01/uv-cache`: dung lượng danh nghĩa khoảng 4,87 GiB nhưng phần lớn dùng hard link với môi trường đã cài; lượng đĩa trống tăng ít hơn nhiều.
- Xóa thư mục `__pycache__` trong workspace: bytecode Python được sinh lại khi chạy.
- Xóa file tải tóc dở `runs/hair-cc0-selection-01/hair01_cc0.partial` (0 byte).
- Đo đĩa trống trước khoảng 5,14 GiB, sau khoảng 5,95 GiB. Chênh lệch quan sát khoảng 0,81 GiB, có thể chịu ảnh hưởng hoạt động khác trên máy.

Giữ nguyên code, input, weights, virtualenv, GLB, Blender, báo cáo và các lần thử. Không chạy test vì chỉ dọn dữ liệu cache; chạy test sẽ tạo lại bytecode.

## Nơi chiếm chỗ cần phân loại trước khi dọn tiếp

| Thư mục | Dung lượng danh nghĩa gần đúng |
|---|---:|
| runs/prod-win-01 sau xóa uv-cache | 6,87 GiB |
| .venv-rdf | 5,17 GiB |
| .venv-triposr | 5,03 GiB |
| runs/prod-win-02 | 2,74 GiB |
| vendor | 1,81 GiB |
| runs/headapi-production | 1,75 GiB |
| runs/assembly-cohesion-01 | 1,58 GiB |
| runs/head-assembly-01 | 1,43 GiB |

Các số trên cộng kích thước file, không đồng nghĩa dung lượng vật lý có thể thu hồi do hard link. Không xóa môi trường/model cũ chỉ dựa vào tên thư mục. Nếu cần thu hồi nhiều GB, kiểm tra các môi trường thử đã bỏ và các bản xuất trùng checksum trước; tránh làm hỏng đường dẫn trong trang review và provenance.

Chi tiết thao tác: `cache-cleanup-20261104.json`, `bytecode-cleanup-20261104.json`.
