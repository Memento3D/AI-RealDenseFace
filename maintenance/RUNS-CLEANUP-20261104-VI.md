# Dọn dữ liệu trùng trong runs

Đã rà soát các file từ 8 MiB trở lên trong `runs`, loại trừ môi trường Python, thư mục nguồn vendored, weights và cache. Các file có cùng kích thước được so SHA-256 trước khi chọn gộp.

Kết quả: 41 nhóm, 80 bản sao được thay bằng hard link, kiểm tra lại SHA-256 từng file. Giữ nguyên đường dẫn, nội dung, model, trang review, nhật ký và thông tin nguồn. Mỗi bước có bản sao tạm để phục hồi nếu tạo link/kiểm tra hash thất bại; bản sao tạm được xóa sau khi kiểm tra đạt.

Dung lượng bản sao gộp: khoảng 3,85 GiB. Dung lượng đĩa trống quan sát tăng từ khoảng 5,95 lên 9,81 GiB (có thể thay đổi do tiến trình khác). Chưa xóa các lần thử có output khác nhau hoặc môi trường AI.

**Lưu ý cho lần làm tiếp:** các file gộp hiện chia sẻ dữ liệu qua hard link. Không sửa đè nội dung trực tiếp lên các artifact cũ. Tiếp tục xuất vào thư mục attempt mới như quy định; nếu cần sửa file cũ, tạo bản sao độc lập trước. Xóa một đường dẫn hard link không xóa dữ liệu ở các đường dẫn còn lại.

Chi tiết nguồn–đích và checksum: `run-dedup-actions-20261104.json`. Danh sách ứng viên ban đầu: `run-duplicates-20261104.json`. Không đưa artifact người dùng vào Git.

Architecture boundaries: PASS. Kết quả unittest lưu tại `run-cleanup-unittest-20261104.log`.
