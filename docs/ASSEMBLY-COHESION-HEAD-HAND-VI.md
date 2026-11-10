# Phối hợp gáy, cổ áo và tay–mũ

Mở `runs/assembly-cohesion-01/review.html`. Bản mới để xem là `attempt-06/contact/assembly.blend`; giữ nguyên các lượt trước để đối chiếu. Chưa thay API hoặc chạy inference ảnh mới.

## Nguyên nhân xác định được

Template là lưới liên thông và vùng tay tiếp xúc với mũ cũ không tách thành object độc lập. Cách chọn vùng đầu bằng hộp trước đây cắt qua tay: body có một đường hở chạy từ cổ áo lên vùng ngón tay. Một số mặt của mũ cũ còn dính giữa các ngón.

Đầu Reconstruction được xuất với nhiều đỉnh trùng tọa độ do UV. Thiếu normal mượt chung qua các đường UV khiến phía sau đầu lộ từng mặt tam giác. Vấn đề này khác với thiếu dữ liệu ảnh phía sau đầu; làm mượt normal không khôi phục chi tiết da đầu thật.

Lòng mũ cũ còn quá chật theo phép kiểm tra envelope: 53 trong 933 đỉnh mẫu phía trên vành nằm ngoài profile lòng mũ. Những đỉnh này có thể là bản sao do UV; không phải 53 điểm độc lập được đo trên người.

## Thay đổi trong bản review cuối

1. Trở lại GLB gốc đã pin để khôi phục tay và cổ tay, thay vì tiếp tục vá trên tay đã bị cắt. Chọn vùng đầu theo khoảng cách geodesic có trọng số màu trên mesh, từ các tập seed theo vị trí của template này. Tách thêm vật liệu mũ cũ ở vùng tiếp xúc tay. Đây là policy riêng cho asset tốt nghiệp hiện có, không phải semantic segmentation tổng quát.
2. Giữ nguyên tọa độ/topology/UV/atlas của đầu; bổ sung normal có trọng số diện tích, chia sẻ qua các đỉnh trùng ở đường UV. Đầu không được subdivide hoặc bóp lại hình mặt trong lượt này.
3. Tạo dải nối giữa 459 đỉnh viền cổ thân và 62 đỉnh đáy cổ đầu bằng zipper gồm 521 tam giác. Hai phía dùng tọa độ biên gốc, không làm tròn tọa độ đầu ra. Da/màu ở dải nối là ước lượng từ vùng cổ nguồn.
4. Bổ sung bề mặt ở những vùng tay tiếp xúc mũ cũ đã bị hở sau khi tách. Những vùng này là geometry/màu suy ra từ lân cận, không phải khôi phục chính xác phần da bị che. Ngón tay vẫn cần được đánh giá hình dáng trực tiếp.
5. Dịch cục bộ vùng tay/cẳng tay với trọng số chuyển tiếp, khóa vùng thân và tay bên kia. Giới hạn điều chỉnh và kiểm tra không đảo hướng tam giác. Lượt cuối có dịch chuyển lớn nhất khoảng 0,0184 đơn vị làm việc; không quy đổi thành kích thước người đã đo.
6. Đưa dây từ mũ về vùng tay và điểm tiếp nối tua mũ còn giữ từ template; không thêm một tua thứ hai. Lòng mũ nới XY khoảng 13,05%, giữ nguyên form thành thẳng và mặt vuông.

## Bằng chứng và phạm vi

`runs/assembly-cohesion-01/audit-final.json` kiểm tra checksum, head geometry/UV/atlas, normal nhúng trong GLB và vị trí các đỉnh thân được khóa. Việc xuất body khớp trường dịch chuyển đã khai báo; không có thay đổi thân ngoài trường đó.

Candidate `restored` là control đã khôi phục nhưng chưa điều chỉnh tay: còn 50 đỉnh nằm trong hộp mặt mũ và 135 tam giác có hộp bao chồng lấn. Không chọn bản này để demo tiếp xúc tay–mũ.

Candidate `contact` có 0 đỉnh tay nằm bên trong hộp mặt mũ và 0 tam giác tay/body có hộp bao chồng lấn mặt mũ trong phép kiểm tra này. Đây là bằng chứng cục bộ tay–mặt vuông mũ; không suy rộng thành mọi va chạm của cả scene đều đã được kiểm tra.

Envelope lòng mũ sau chỉnh có 0 đỉnh mẫu nằm ngoài profile ellipse được kiểm tra. Chưa có chứng nhận clearance đầy đủ theo mọi tam giác hoặc cấu hình máy in.

Chẩn đoán sau hàn tọa độ ở ngưỡng 1e-7 còn 75 cạnh hở trong phần core được kiểm tra, 0 cạnh có hơn hai mặt. Vì vậy không ghi nhận model kín hoàn toàn hoặc print-ready. Mối nối cổ và các vùng vá tay chưa qua kiểm tra tự giao đầy đủ.

Sáu tests mới cho normal, vùng khóa thân/tay, biên UV, zipper cổ, độ chính xác biên và envelope mũ PASS; architecture PASS. Full suite: 562 tests, 1 failure, 4 errors, 1 skipped; các lỗi workspace resume / hair manifest / im2 còn ghi trong `unittest-verified.log`.

## Cách xem

Trang review cho đổi giữa phía sau đầu, góc sau, cận tay, cạnh tay, mặt trước, toàn thân và geometry xám. Các cặp cận tay đã render lại cùng camera; không dùng việc đổi góc máy để che vùng lỗi.

Trong Blender có các component đầu, thân, lòng/mặt vuông mũ, dây, dải cổ và các patch tiếp xúc. Nên kiểm tra cả mặt sau của ngón tay và vùng cổ áo, không chỉ ảnh chính diện.

## Giới hạn cần giữ rõ

Nguồn tay đã gắn với mũ cũ, nên bề mặt bị che không đủ dữ liệu để khôi phục tay chính xác hoàn toàn. Một số đầu ngón và nếp gấp còn cần chỉnh asset template trong Blender. Không coi bản vá tự động là giải pháp giải phẫu tổng quát.

Da phía sau đầu vẫn là texture ước lượng từ các thí nghiệm trước; normal mượt chỉ cải thiện shading. Độ giống cá nhân chưa được xác minh lại và không thay hệ số identity trong lượt này. Đây là bản kiểm chứng phối hợp asset, chưa phải model hoàn thiện để sản xuất hoặc kết quả API đã đổi.
