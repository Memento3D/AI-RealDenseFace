> Cập nhật 2026-11-05: API fit theo job đã được nối và kiểm chứng. Tài liệu trạng thái/hướng dẫn hiện hành: [MVP-ASSET-STORAGE-AND-AI-API-HANDOFF-VI.md](MVP-ASSET-STORAGE-AND-AI-API-HANDOFF-VI.md). Nội dung dưới đây mô tả mốc kiểm tra trước đó.

# Bộ asset được chọn cho MVP preview v1

Ba asset: graduation-cap@v1, hair-side-part@v1, glasses-round@v1. Kính có lựa chọn null/Không kính; mũ và tóc loại trừ nhau. Mặc định mũ + không kính. Kính mẫu 01, tóc khác và body/dáng khác không thuộc phạm vi v1.

Đã render và kiểm tra bốn tổ hợp trên baseline e0a57466-a84b-4609-a18e-eeb95f749885. Không thay mặt, cổ hoặc body của baseline. Đây là lựa chọn phạm vi MVP preview, không phải phê duyệt khả năng in hoặc bảo đảm fit mọi ảnh.

## File dùng chung

catalog.json và assets/<id>/v1/{preview.glb,thumbnail.png,manifest.json}. Thumbnail được render từ đúng phụ kiện đã xử lý, không có mặt/body user. Bằng chứng nguồn CC0 kính/tóc được giữ riêng trong license-evidence.html; nguồn body/mũ là template dự án, quyền phân phối chưa được audit.

Preview mũ đã giảm atlas từ 4096 xuống 1024, loại 227 mặt màu da và giới hạn tua rua vào vùng thân tua để bỏ phần tay lẫn. Giữ lỗi bề mặt nhỏ ở mép board để tối ưu sau. Đây là biến thể preview, không thay thế nguồn độ phân giải cao, không dùng làm master in.

## Tọa độ và fit

GLB theo glTF Y-up. Các asset dùng chung cần fit theo từng đầu. Mũ/tóc dùng điểm neo mắt của baseline để chuẩn hóa origin; kính dùng convention nguồn đã có profile. Canonical GLB không có nghĩa là có thể cộng thẳng vào scene cho mọi user.

Kính có chỉnh hình học càng theo đầu, nên một ma trận affine là chưa đủ để mô tả kết quả fit. BE/AI cần lưu fitResultVersion tham chiếu fitted GLB riêng hoặc thực thi lại fitter có version. Fitted GLB đã ở tọa độ của base, ma trận scene tham chiếu là identity; tuyệt đối không căn giữa phụ kiện độc lập.

## Review riêng và bước tiếp theo

validation/customization/mvp-selected-preview-v1/index.html chứa ảnh có mặt user và bốn scene ghép, không đưa vào catalog chung. scene-manifest.json ở đó chỉ áp dụng một baseline. Asset library vẫn published=false do chưa upload/đăng ký catalog BE.

Lưu B2 vào library-drafts/mvp-selected-preview-v1/ để bảo toàn cấu trúc, upload thumbnail chính thức lên Cloudinary ở bước publish. Neon lưu catalog/version và key qua importer BE; không lưu binary GLB. Chưa sửa cấu hình API AI hoặc nối FE/BE trong bước lựa chọn asset này.
