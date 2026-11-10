# Chỉnh tỷ lệ đầu và form mũ sau review

Review: `runs/graduation-assembly-04/review.html`.

## Đã sửa

Mũ ở bản trước có profile sqrt(1−t²), khiến thành mũ khép thành vòm như một chiếc bát úp. Bản mới dùng thành vải gần thẳng: bán kính trên còn 96,5% bán kính dưới; chiều cao thành ngoài giảm từ 0,051 xuống 0,034 trong hệ đơn vị thiết kế. Đỉnh thành mũ nằm sát mặt vuông phía trên. Đây là geometry tham số, chưa phải mô phỏng vải hoặc mũ lấy từ ảnh.

Mặt vuông giảm từ bề rộng cố định 0,23 sang kích thước theo envelope đầu, có ngưỡng dưới 0,16. Cả hai candidate hiện có mặt vuông rộng 0,16. Dây được nối qua điểm thiết kế gần tay đang giơ để xem bố cục; chưa chứng nhận điểm tiếp xúc hay va chạm của bàn tay.

Hai tỷ lệ đầu mới:

| Candidate | Scale so với bản căn đầu ban đầu | So với compact trước đó | Vị trí mắt thấp hơn compact trước |
|---|---:|---:|---:|
| balanced | 0,90 | Nhỏ hơn khoảng 4,3% | 0,008 đơn vị làm việc |
| slim | 0,86 | Nhỏ hơn khoảng 8,5% | 0,013 đơn vị làm việc |

Chỉ dùng scale đồng nhất, yaw và tịnh tiến để thay tỷ lệ đầu–thân. Tỷ lệ giữa các đặc điểm mặt được giữ theo geometry nguồn của cùng biến thể hàm. Cả hai dùng cùng thử nghiệm giảm góc hàm 25% đã qua guard ở lượt trước; không fit lại identity hoặc thay model/texture trong lượt này.

Việc hạ vị trí đầu và giữ mặt phẳng cắt đáy cổ giúp giảm chiều dài cổ lộ ra. Không có phép hàn cổ hay Boolean union mới.

## Kiểm chứng

- Audit file xuất xác nhận thân giữ nguyên geometry/topology/UV, atlas mặt giữ pixel và từng phần mũ có topology kín.
- Năm tests pose/cap PASS. Test mới buộc tỷ lệ bán kính trên/dưới của thành mũ >0,9 để tránh quay lại profile bán cầu.
- Architecture PASS; full unittest được chạy lại, log cạnh review. Không báo toàn suite PASS nếu log còn lỗi.
- Hai Blender scenes và GLB được xuất vào thư mục mới, không thay các attempt trước hay API.

Full suite cuối: 556 tests, 1 failure, 4 errors, 1 skipped. Failure ở mocked workspace resume; errors ở hair manifest thiếu file, im2 yêu cầu Linux và thiếu pyhocon như các lượt trước. Log: `runs/graduation-assembly-04/unittest.log`.

Ảnh trước/sau mặt trước và toàn thân dùng cùng camera/render settings. Góc nghiêng mới nâng camera lên để thấy mặt vuông của mũ; không coi so sánh góc nghiêng với ảnh trước là cùng camera. Chế độ photo-only là diagnostic shader từ lượt trước, không phải texture mới.

## Chưa giải quyết

Góc nhìn cao vẫn thấy khe quanh cổ áo và phần template còn sát phía sau cổ. Đây là hậu quả của bước tách thân trước, cần chuẩn bị vòng nối sạch và xử lý geometry riêng. Bản này vẫn là thử nghiệm tỷ lệ/form mũ, chưa phải asset kín hoặc đã xác nhận in được.

Hình dạng khuôn mặt nguồn và độ giống cá nhân chưa được tối ưu lại trong lượt này. Thu nhỏ toàn bộ đầu không đồng nghĩa làm má hoặc hàm thon đi theo ảnh gốc. Nếu sau review vẫn cần chỉnh hình mặt, phải đối chiếu fitting đầu trên các ảnh nguồn thay vì tiếp tục scale hoặc bóp ngang để vừa thân.
