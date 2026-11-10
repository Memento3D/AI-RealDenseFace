# Thử ghép đầu Reconstruction với thân Production

## Mục tiêu và phạm vi

Thay toàn bộ vùng đầu template bằng đầu trọc đã dựng trong Head Reconstruction, thay cho vá một mảng mặt vào đầu của người mẫu. Đây là thử nghiệm offline dùng lại đầu input04 đã kiểm chứng trước đó; không chạy inference mới, không sửa API, không gộp các môi trường SDK hoặc sửa repo Production gốc.

Adapter mới là `tools/headapi/head_assembly_trial.py`. Nó đọc GLB thân từ bản runtime Production đã pin, đọc GLB đầu và manifest từ Head Reconstruction, rồi tạo GLB/Blender để review. Đây là kết nối qua artifact và contract tọa độ, chưa phải pipeline nhận ảnh mới cho mọi người.

## Kết quả mở được

Mở `runs/head-assembly-01/review.html`. Hai biến thể cuối ở `attempt-04/neck-low` và `attempt-04/neck-high`, mỗi biến thể gồm:

- `assembly.blend`: scene có đầu và thân riêng, camera và đèn; có thể chỉnh tiếp trong Blender.
- `assembly.glb`: bản ghép hai component `Body_Template` và `Personalized_Complete_Head`.
- `body.glb`: phần thân sau tách thử.
- `removed-template-head.glb`: phần bị loại để kiểm tra vùng cắt, chưa phải một component đầu được phân đoạn ngữ nghĩa sạch.
- Năm ảnh render: toàn thân xám, mặt trước xám, góc nghiêng xám, bên xám và mặt trước có màu.

Đầu không còn bị ép vào đường bao mặt của template, không thêm kính/hair-cap của template vào đầu cá nhân. Kính từ ảnh chưa được tái dựng. Bàn tay và quần áo dùng vật liệu gốc; chưa tự đồng bộ màu da tay với đầu mới.

## Cách thực hiện

Template có 1,5 triệu tam giác. Sau hàn tọa độ làm tròn ở bước khảo sát, toàn bộ là một thành phần liên thông; không thể chỉ dùng Separate by Loose Parts để có ngay các bộ phận ngữ nghĩa.

Tám landmark 3D nhúng trên FLAME được dùng để fit similarity gồm xoay, tịnh tiến và scale đồng nhất. Không kéo mặt theo từng điểm để khớp đầu mẫu. Sai số landmark khoảng 0,003–0,009 theo hệ đơn vị làm việc; đây không phải độ chính xác tái dựng, không phải kích thước người được đo.

Hệ làm việc Z-up/front -Y và glTF Y-up được chuyển đổi rõ bằng matrix. Các vùng cắt thân hiện được khai báo riêng cho template này. Bản low bắt đầu ở Z=.775, high ở Z=.785; chúng chưa phải thuật toán phân đoạn đầu tổng quát.

Vùng chân cổ loe của FLAME được thu gọn bằng mask neck của model đã pin, sau đó cắt đáy ở Z=.765 bằng nội suy UV. Các đỉnh không thuộc mask cổ được kiểm tra giữ nguyên trước bước cắt đáy. Cổ mới đang lồng vào cổ/thân cũ để xem tỷ lệ, chưa hàn biên hay Boolean union.

## Kiểm chứng và các bản bị loại

- Attempt 01: giữ lại để ghi nhận lỗi quy ước trục glTF; không dùng làm mẫu cuối.
- Attempt 02: cắt hộp quá rộng, mất tay và cổ áo; đã bị loại bằng visual audit. RAM guard cũng báo lỗi muộn dù file export đã ghi xong; không coi lượt guarded đó là PASS.
- Attempt 03: vùng tách hẹp hơn giữ tay, nhưng phần chân cổ đầu mới còn loe.
- Attempt 04: thu gọn riêng cổ bằng mask, đã render cả hai biến thể và kiểm tra file xuất độc lập.

`runs/head-assembly-01/audit.json` xác nhận hai biến thể attempt 04 giữ đúng tọa độ và UV của các tam giác thân được giữ lại. 9.213 tam giác đầu có mọi đỉnh trên Z=.82 giữ geometry với sai số xuất float32 dưới 1e-7 và UV giữ nguyên. Pixel atlas đầu nhúng trong GLB giống nguồn. Kiểm tra này có phạm vi rõ; không suy rộng thành mọi tam giác đầu/cổ không thay đổi.

Bốn test mới cho volume selection và bảo vệ đỉnh ngoài cổ PASS. Full unittest và architecture được chạy theo AGENTS.md; kết quả/log ở thư mục thử nghiệm. Các lỗi suite cũ phải được báo riêng, không coi toàn suite PASS.

## Những gì cần sửa tiếp trước khi gọi là template hoàn chỉnh

1. Kiểm tra và đánh dấu sạch đường tách sau cổ, chân tai và mảnh template còn sát cổ áo. Màu render cho thấy dải geometry sau cổ cần phân loại lại, chưa kết luận đó là phần đầu hay phụ kiện/cổ áo cần giữ.
2. Tách dây tua/mũ và vùng da tay thành accessory/material riêng; hiện vẫn còn dây tua treo cạnh tay theo pose gốc.
3. Chọn một vòng nối cổ ở thân và một vòng ở đầu, nối có kiểm soát rồi kiểm tra tự giao, lỗ và độ dày. Không dùng bản ghép hai mesh hiện tại để khẳng định in được.
4. Giải quyết đường chuyển màu ở cổ và vùng da hở sau khi hình khối được chấp nhận. Texture da đầu/ vùng khuất vẫn là ước lượng từ thí nghiệm cũ, không phải chi tiết quan sát đầy đủ.
5. Kiểm tra thêm các đầu khác trước khi tự động hóa nối vào API.

Đề xuất giữ `neck-low` làm điểm bắt đầu chỉnh template: đó là lựa chọn để review, chưa phải bản đã được người dùng duyệt hoặc đã kiểm chứng mối nối. Có thể mở `.blend` và ẩn/hiện từng component để quyết định vùng cần chỉnh; không phải sửa file gốc.
