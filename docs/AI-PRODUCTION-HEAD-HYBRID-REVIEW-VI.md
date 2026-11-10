# Kết hợp Head Reconstruction và AI Production — 04/11/2026

Đã thử cùng các ảnh `input/04` và kiểm thêm `input/03`, giữ outputs cũ. Trang đối chiếu: [runs/prod-win-02/review.html](../runs/prod-win-02/review.html). Báo cáo đầy đủ: [README thử nghiệm](../runs/prod-win-02/README.md).

## Kết quả

Hai bộ ảnh đều chạy Production tới GLB, quality gate PASS, kết quả needs_visual_review. Input04 có bốn candidate: Production gốc, direct texel giữ tone gốc, direct texel với offset ±12 RGB, và geometry MICA kết hợp UV Head Reconstruction với offset ±12. Cả bốn có cùng geometry MICA theo audit positions/indices/node transforms; khác nhau về nguồn/cách chuyển màu.

Đã tìm thấy hai vấn đề trước khi cần thay model: vertex color sampling trên grid patch 0,001 working units rồi nội suy ra atlas có thể làm mất chi tiết; offset hòa da gốc tới khoảng +51 RGB có trọng số luminance có thể làm nhạt mặt. Hai thay đổi được tách thành controls riêng, không chỉ đổi nhiều thứ rồi tuyên bố model tốt hơn. Direct/bounded cũng được kiểm thêm trên input03.

Bản MICA + UV RDF qua kiểm hình học/texture/GLB. Giữ source provenance/confidence: khoảng 16,2% patch dùng nguồn inferred/support thấp, 0,28% fallback; không khôi phục chi tiết chưa thấy. Chưa có đánh giá ground truth để chọn model có độ giống tốt nhất.

## Geometry RDF chưa được ghép thành công

Topology hai nguồn trùng 5.023 vertices và 9.976 face indices, nhưng mapping geometry RDF có 59/7.019 điểm core chạm bề mặt sau/ngoài vùng mặt theo guard và template prior. Đã giữ nguyên guard, không xuất model thành công cho nhánh đó. Diagnostic và failed logs nằm trong thư mục review. Cần chuẩn hóa coordinate/pose và giới hạn front facial surface, xử lý nơi thiếu support với provenance rõ trước khi thử lại.

## Hướng kiến trúc đề xuất

Giữ Producer Head Reconstruction cho geometry/UV/visibility/masks/provenance; giữ Consumer Production cho body, trang phục, template binding và mesh validation. Contract giữa hai bên cần khai báo model pins, topology, axes/units, pose/frame, texture UV, source support và input hashes. Không merge tất cả SDK vào một env hoặc tự thay model âm thầm.

Trước mắt, dùng các candidate để review ảnh chính diện/nghiêng và source texture; giữ baseline Production. Nếu chọn MICA + UV RDF, tích hợp bằng adapter có policy version riêng. Bước tiếp của geometry là sửa source-surface contract, không tăng ngưỡng nhận mặt sau. Các candidate trong lượt này chỉ cải thiện appearance; chất lượng mặt in đơn sắc và khả năng in chưa được xác nhận.

## Trạng thái và kiểm chứng

- Chưa đổi default của launcher, API hoặc web. Tất cả thay đổi màu nằm trong worker thử nghiệm và các GLB mới.
- Candidate màu đã qua kiểm pins, geometry bằng baseline, atlas nhúng đúng và ba texture PBR của tượng giữ pixel. Link review và preview modes đã kiểm tra tồn tại.
- Năm tests toán raster/provenance/UV origin PASS, architecture PASS. Suite chính: 535 tests, 1 failure, 4 errors, 1 skipped (các nhóm lỗi đã có); log nằm ở thư mục thử nghiệm.
- Không fine-tune, tải weights mới hoặc lên cloud; worker chạy tuần tự với RAM guard. RDF fit là cached baseline cùng input04, không phải kiểm thử độc lập chưa từng xem.
- Báo cáo comparison ghi PARTIAL_REVIEW_ONLY vì nhánh geometry RDF chưa qua guard; không báo toàn bộ thí nghiệm thành công.
