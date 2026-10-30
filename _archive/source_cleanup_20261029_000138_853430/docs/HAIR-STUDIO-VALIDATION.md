# Hair Studio V2.3 — kiểm chứng thực tế

- 87 unittest Python đạt, gồm 11 kiểm thử mới cho Studio và 76 kiểm thử nền V2.2/HeadLab có trong gói.
- 3 kiểm thử JavaScript đạt cho last-request-wins, xóa mẫu đang nạp và thay revision.
- 8 tình huống trình duyệt WebGL thực tế đạt trên CPU/Linux, Chromium 131 với SwiftShader: mở đầu và tóc V2.2, đổi mẫu có metadata, thêm GLB không cần chỉnh HTML, thay file, xóa mẫu đang dùng, xuất GLB, chế độ xám và layout 390px.
- Không có page error trong lần thử trình duyệt. Không suy ra tương thích với mọi GPU/browser Windows.
- Đổi mẫu giữ nguyên UUID node đầu, checksum GLB đầu và camera. Xuất GLB qua UI giữ nguyên từng phần tử buffer vị trí đầu; chỉ thêm mesh tóc và transform preview.
- Đã thử thêm file GLB mới và sidecar ngay trong assets/hair khi trang đang mở; trang tự phát hiện qua polling, không dùng nút refresh.
- Đã thử đổi nội dung cùng file (revision mới), rồi xóa file đó; đầu được giữ nguyên.
- Code và demo được kiểm tra ZIP CRC và SHA sau giải nén sạch. Vendor Three.js 0.160.1 đóng gói cục bộ cùng LICENSE.
- Chưa thực thi PowerShell trên Windows trong môi trường này. Chưa xác minh fit/in cho mọi mẫu tóc. Studio chỉ xuất preview GLB, không tạo cam kết in được.

Các ảnh chụp trình duyệt chỉ phục vụ kiểm tra nội bộ, không được đóng gói. Demo chứa model đầu mẫu 04 riêng biệt với gói mã.
