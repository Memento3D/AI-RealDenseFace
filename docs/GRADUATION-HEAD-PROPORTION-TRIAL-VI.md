# Phối hợp đầu, thân, mũ và kiểm tra biểu cảm

Mở `runs/graduation-assembly-03/review.html`. Đây là thử nghiệm offline tiếp nối bản ghép đầu hoàn chỉnh, chưa đưa vào API và chưa phải kết quả độ giống đã được người dùng duyệt.

## Những thay đổi đã thực hiện

- Bỏ pitch/roll mà phép căn theo đầu template đưa vào; dùng yaw, scale đồng nhất và vị trí neo theo trung bình bốn mốc mắt. Không kéo từng đặc điểm khuôn mặt để khớp người mẫu.
- So sánh đầu `upright` với `compact`: compact có scale bằng 94% bản căn trước và điểm neo mắt thấp hơn 0,005 đơn vị làm việc. Đây là tham số thiết kế để review, không phải tỷ lệ giải phẫu đã hiệu chuẩn.
- Thêm mũ tốt nghiệp thành các component độc lập: `Cap_Cup`, `Cap_Board`, `Cap_Cord`. Mũ này được dựng tham số từ envelope da đầu, không phải mũ tái dựng từ ảnh hoặc mũ template đã được phân đoạn tự động sạch.
- Giữ đầu và thân riêng, giữ geometry/UV của thân. Pixel atlas mặt không đổi; không làm sáng toàn khuôn mặt để ép khớp màu template.
- Thêm biến thể `compact-jaw-bounded` để thử giảm hé miệng đã có trong nguồn RDF. Model được decode lại từ đúng identity và weights cũ, không training/fine-tune hoặc thay hệ số identity.

Mỗi biến thể có GLB và Blender scene để mở, ẩn/hiện đầu, thân và ba phần mũ. Có render mặt trước, góc nghiêng, toàn thân, geometry xám và diagnostic texture.

## Vì sao chưa được nói là đã khôi phục nét cá nhân

Audit lượt trước xác nhận 9.213 tam giác vùng đầu phía trên Z=.82 và atlas giữ nguyên khi ghép. Điều này cho thấy phần được kiểm tra không bị biến thành mặt template trong bước ghép; nó không chứng minh nguồn RDF vốn đã giống người trong ảnh.

Trong ảnh input04, môi gần khép; đầu RDF hiện có biểu cảm hé miệng. Tư thế đầu, biểu cảm, shader bổ sung ánh sáng lên texture đã có ánh sáng trong ảnh, và việc bỏ tóc đều có thể ảnh hưởng cảm nhận. Chưa định lượng được đóng góp của từng yếu tố vào độ giống.

Biến thể hàm là thử nghiệm biểu cảm có kiểm soát, không phải identity fitting mới. Không dùng mũ hay việc texture giữ nguyên làm bằng chứng độ giống đã cải thiện.

## Thử nghiệm hàm thực tế

Worker `runs/graduation-assembly-01/jaw_trial.py` dùng môi trường RDF hiện có và đúng hai model pins. Trước thử nghiệm, decoder phải tái tạo geometry nguồn với sai số tối đa 1e-6. Giữ nguyên 300 hệ số identity và 100 hệ số expression, chỉ thay ba thành phần axis-angle của khớp hàm.

Lượt giảm góc hàm 50% bị loại vì dịch chuyển lớn nhất bằng 1,22% chiều cao đầu, vượt ngưỡng thử nghiệm 1%. Backtracking giữ lại 75% góc hàm ban đầu: dịch chuyển lớn nhất khoảng 0,61% chiều cao đầu, không đảo hướng tam giác. Không có tam giác suy biến nguồn bị loại khỏi lần đếm này.

Chẩn đoán khoảng mở môi theo embedding FLAME giảm từ khoảng 0,0317 xuống 0,0156 khi chuẩn hóa bằng khoảng cách hai khóe mắt của model. Đây chỉ là chỉ số geometry của model, không phải sai số đo trên ảnh hay điểm nhận diện. Chưa kiểm tra self-intersection và chưa chứng minh việc giảm hé miệng giống người hơn.

Đây là decode biểu cảm từ fit đã lưu, không phải inference ảnh mới. Dữ liệu đầu vẫn là input04 đã dùng trong các lần thử trước; chưa có đánh giá trên người chưa từng dùng để chỉnh tham số.

## Màu và ánh sáng

Ảnh `color-*` dùng material PBR của GLB. Ảnh `photo-only-diagnostic` chỉ hiển thị atlas mặt bằng emission để so sánh tác động của chiếu sáng thêm. Diagnostic không sửa texture nguồn và không được thay vào material master khi lưu Blender. Đây không phải albedo đã tách sáng hay màu da được hiệu chuẩn.

Các bản 01/02 được giữ lại để đối chiếu pose và thiết kế mũ. Bản 03 dùng vật liệu mũ đen nhám hơn; không thay model pins.

## Kiểm chứng và giới hạn

`runs/graduation-assembly-03/audit.json` xác nhận thân giữ nguyên vị trí/topology/UV và atlas mặt nhúng giữ pixel ở cả ba biến thể. Các phần mũ riêng có topology kín; điều đó không chứng minh cả tổ hợp in được. Chưa chứng nhận clearance mũ–da đầu, va chạm dây với tay, hàn cổ, tự giao hoặc độ dày cho resin.

Còn vùng cổ/template sát cổ áo cần làm sạch. Dây tua từ mẫu gốc vẫn còn và chưa được tách semantic hoàn chỉnh. Màu tay vẫn thuộc template, chưa đồng bộ màu với đầu. Vì vậy đây là các candidate để kiểm chứng, chưa phải asset hoàn chỉnh dùng sản xuất.

Bốn test pose/cap PASS; architecture PASS. Full suite được chạy lại và log lưu cạnh review; không coi toàn suite PASS nếu còn các lỗi ở nhóm hair/im2/workspace cũ.

Lượt cuối: 555 tests, 2 failures, 4 errors, 1 skipped. Ngoài lỗi workspace resume và nhóm im2/manifest đã ghi nhận trước, full suite còn gặp lỗi cache revision ở hair studio. Chạy lại riêng 11 tests hair studio thì PASS; chưa kết luận nguyên nhân hoặc sửa nhóm đó trong lượt này. Log riêng `hair-studio-recheck.log` được giữ cạnh log tổng.

## Bước tiếp theo sau review

Chọn tỷ lệ/tư thế và biểu cảm dựa trên ảnh gốc, rồi giữ cố định các tham số ghép đó. Tiếp tục đánh giá shape của đầu độc lập trên ba ảnh nguồn: viền hàm, độ đầy má, mũi, môi và mắt, thay vì tiếp tục chỉnh thân để che sai lệch. Nếu hình đầu vẫn thiếu nét cá nhân thì cần quay lại bước fitting đầu; không giải quyết bằng cách tăng độ nét texture hoặc bóp khuôn mặt theo body.
