# Khung đánh giá Pipeline V2 — bằng chứng trước khi kết luận

## Phạm vi và mục tiêu

Sản phẩm hướng tới: tượng tổng thể 80–100 mm gồm đầu, thân và đế. Batch hiện
tại chỉ xử lý đầu+cổ, mặc định 22 mm trước khi thêm tóc. Nguồn là workspace
đầu đã có; V2 chưa thay thế bước tái dựng đầu từ ảnh.

Ngắn hạn: thư viện tóc, khớp có giới hạn, màu da thống nhất, phối hợp mắt–miệng,
kiểm tra hình học và thử in. Dài hạn: tăng cá nhân hóa khi có dữ liệu và tiêu chí.

## Quy tắc đọc bằng chứng

1. Quan sát từ ảnh không tự xác định nguyên nhân hình học.
2. Ghi riêng điều đo được, phép đo lấy mẫu/proxy, giả thuyết và phần chưa kiểm chứng.
3. Không dùng một điểm số chung để bù giữa lỗi in, độ giống và thẩm mỹ.
4. Mesh kín theo cạnh chưa đủ: kiểm tra cả liên kết các mặt quanh mỗi đỉnh.
5. Các ngưỡng hiện là thử nghiệm kỹ thuật; chưa có hiệu chuẩn theo tập người dùng.

Ví dụ cần sửa cách kết luận: tóc trông đen phẳng có thể vẫn có khối hình học;
răng cưa có thể đến từ mask, mesh hoặc render; lỗi ảnh sau có thể do camera;
cổ cắt phẳng có thể là giao diện ghép thân chứ không phải lỗi.

## Bước 0: đầu vào và camera

- Kiểm tra file, checksum model, topology đầu nguồn và thông tin đơn vị.
- Ghi kích thước ảnh và hỗ trợ semantic da/mắt/tóc.
- Ghi sai số chiếu nội bộ camera, không gọi đó là hiệu chuẩn độc lập.
- Ghi điều chưa biết: biểu cảm giữa các ảnh, cùng trạng thái tóc, vùng bị che.
- Mẫu tóc ở phía sau là thiết kế mẫu; V2 không dùng ảnh back để đo chân tóc gáy.

Sai số chiếu nội bộ nhỏ chỉ cho biết camera gần dữ liệu chiếu đã lưu; không
chứng minh sọ hoặc tóc đúng ngoài đời. Pipeline không che giấu lỗi nguồn bằng
cách sửa mũi, hàm hoặc mắt theo một tỷ lệ làm đẹp chung.

## Năm góc nhìn

| Góc nhìn | Điểm có thể kiểm tra | Điểm yếu/rủi ro cần thu thập | Bằng chứng cần có |
|---|---|---|---|
| Khách hàng | Xem trước trên đầu cá nhân, đổi mẫu tóc | Không nhận ra mình; tóc đẹp nhưng sai sở thích; preview khác vật thể | So sánh mù độ giống, hài hòa, cảm xúc và ý định đặt in |
| Thiết kế 3D | Liên kết mặt/đỉnh, bề mặt tóc, độ khớp | Pinch, khe hở, tóc che bộ phận quan trọng, biến dạng mất style | Audit mesh và đánh giá nhiều góc |
| AI/CV | Phân đoạn, chiếu camera, đề xuất mẫu | Lỗi camera, dữ liệu che khuất; silhouette không xác định chân tóc | Nhãn tham chiếu, ảnh giữ lại ngoài tối ưu, nhiều người độc lập |
| In 3D | Topology, kích thước, support thử nghiệm | Chi tiết mỏng, liên kết yếu, peel force, resin và hoàn thiện màu | Lớp cắt thực, profile hiệu chuẩn, in và đo vật thể |
| Giải phẫu/biểu cảm | Giữ bề mặt cốt lõi so với đầu nguồn | Đầu nguồn chưa đúng; mắt–mí lệch; biểu cảm giả; mất bất đối xứng cá nhân | Dữ liệu tham chiếu và đánh giá người có chuyên môn |

Đây là năm **góc nhìn đánh giá**, không có nghĩa đã có năm chuyên gia chấm điểm.
Người Việt/Á Đông là nhóm cần được đại diện trong dữ liệu kiểm thử, không phải
lý do áp một bộ tỷ lệ đẹp cố định hoặc ép đối xứng cho mọi người.

## Ba vấn đề chính và tiêu chí hiện thực

| Nhóm | Tiêu chí | Hiện có trong bản V2 | Chưa được chứng minh |
|---|---|---|---|
| A. Tóc | Giữ style, khớp đầu, tránh xâm lấn mặt | Metadata mẫu, affine/cage, overlap, mẫu điểm mặt, giới hạn custom, sửa riêng tóc | Đúng chân tóc thật, độ đẹp, thích nghi toàn bộ dáng đầu |
| B. Mặt | Nét nhận diện, biểu cảm, màu nhất quán | Giữ đầu nguồn; preset má/khóe môi nhẹ; mắt chung hướng; nền da và sRGB/linear đúng quy ước | Giống người độc lập, giải phẫu đúng, màu da vật lý đúng |
| C. Chế tạo | Bề mặt hợp lệ và phù hợp máy/resin | Đọc lại STL; kiểm tra cạnh/đỉnh; phép lấy mẫu tia; support/voxel thử nghiệm | Độ dày tối thiểu chính xác, độ bền, slicer/mã máy, tỷ lệ in thành công |

### Phát hiện từ hai STL nguồn

Kiểm tra cạnh cho thấy cả hai file kín, hướng nhất quán và một thành phần.
Kiểm tra vertex-link bổ sung phát hiện 11256 có 87 đỉnh bị chia nhiều fan mặt;
Euler=-85. Mẫu 21778 có Euler=2 và không có đỉnh fan lỗi trong phép kiểm tra này.
V2 dùng phát hiện đó để kích hoạt sửa tóc, rồi kiểm tra bề mặt cuối sau xuất.

Euler khác 2 không tự chứng minh lỗi: một bề mặt kín có thể có đường hầm hợp lệ.
Không ép mọi kiểu tóc/búi/nơ về genus 0. Việc kiểm tra fan đỉnh mới là bằng
chứng cụ thể về pinch; watertight đơn thuần bỏ sót trường hợp này.

## Q&A A — tóc

**A1. Có cần tóc giống ảnh 100% ở V2?** Không. Đề xuất mẫu theo độ nhất quán
silhouette, cho người dùng đổi style; ghi rõ đó là mẫu, không phải tóc tái dựng.

**A2. Tóc bám đầu và tóc dài xuống gáy có giống nhau?** Không. Vùng tiếp giáp
và khối ngoài là hai phần khác nhau. V2 dùng lòng tóc/neo của mẫu, không chiếu
một mẩu tóc trong ảnh nghiêng thành vùng tóc mọc trên cổ.

**A3. Khớp bằng cách nào?** Co giãn theo kích thước vùng sọ, căn roof/lower
anchor, rồi radial cage có giới hạn. Affine được giữ làm đối chứng nhỏ.

**A4. Có đảm bảo mọi đầu đều khớp một mẫu?** Không. Ngoài giới hạn scale/cage
hoặc xâm lấn mặt thì loại mẫu; không kéo đến khi méo hoặc sửa mặt để vừa tóc.

**A5. Thêm mẫu mới có phá pipeline?** Không nếu tuân contract. Cần chuẩn bị
hướng, đơn vị, cavity và kiểm định một lần. Thay đổi data registry không sửa mã.

**A6. Có thể thay tóc sau khi in?** Chưa. Hiện thay trong phần mềm; STL hợp
đầu–tóc nguyên khối. Tháo lắp vật lý cần interface/dung sai riêng.

## Q&A B — mặt

**B1. Giữ giống hay tự làm đẹp?** Giữ dáng đầu nguồn; biểu cảm nhẹ là preset
tường minh. Không tự đổi sống mũi, hàm, khoảng mắt theo chuẩn đẹp chung.

**B2. Source có phải biểu cảm trung tính?** Không nhất thiết. Nó là hình học
nguồn chưa sửa biểu cảm. V2 không có bằng chứng đã khôi phục neutral expression.

**B3. Hệ thống đã chứng minh nhận ra người chưa?** Chưa. Sai lệch bề mặt cốt
lõi nhỏ sau boolean chỉ chứng minh giữ mesh nguồn tại các điểm lấy mẫu.

**B4. Mắt và miệng phối hợp thế nào?** Mí là hình học nguồn, mắt chung hướng,
má/khóe môi thay đổi nhỏ có giới hạn. Cần đánh giá cảm nhận; không gọi preset
này là mô phỏng cơ mặt hay nụ cười tái dựng đúng giải phẫu.

**B5. Răng được làm thế nào?** Vùng màu liên tục trên phần trong miệng khi có
bằng chứng sáng phù hợp hoặc user chủ động chọn show. Không có răng rời dễ gãy.

**B6. Màu da và đặc điểm riêng?** Nền da từ vùng da semantic, trim luminance,
đề xuất một trong tám màu dự án. Không có hiệu chuẩn ánh sáng/da thật. Vết bớt
và hình xăm là lớp bổ sung sau; hiện chưa tự tạo hoặc bảo toàn chúng.

## Q&A C — chế tạo

**C1. Kín là in được?** Không. Cần manifold cả cạnh và đỉnh, các phép đo chi
tiết, profile vật liệu/máy, lớp cắt và dữ liệu in thực tế.

**C2. Tia kiểm tra có cho độ dày nhỏ nhất?** Không. Chord theo normal là một
proxy lấy mẫu, có thể bỏ sót hoặc đánh dấu vùng không đại diện cho thành chịu lực.

**C3. Sửa mọi mesh được không?** Không. Chỉ sửa tóc một lần trong ngân sách;
kiểm tra thể tích, sai lệch lấy mẫu và bề mặt cuối. Không đạt thì loại.

**C4. Có thể dùng ngưỡng 0.4 mm chung không?** Đây chỉ là ngưỡng thử nghiệm
được lưu trong profile. Chưa phải giới hạn đã đo cho Saturn và resin cụ thể.

**C5. Support tự động đã đủ chưa?** Chưa. Hiện là mô hình và kiểm tra voxel
thô, không kiểm định peel force, độ bền chân support hay lớp máy thực.

**C6. Tách tóc có in nhiều màu được không?** Không tự động. GLB có màu; Saturn
in resin theo vật liệu đang dùng. Hoàn thiện/tách vật liệu là quy trình khác.

## Các mức tự động hóa phải báo riêng

1. Tạo hình và đề xuất mẫu: tự động trong phạm vi đầu nguồn được hỗ trợ.
2. Khớp, sửa có giới hạn, kiểm định hình học: tự động, có thể loại ứng viên.
3. Hướng in/support: tự động thử nghiệm.
4. Slicer/mã máy theo resin hiệu chuẩn: chưa tích hợp.

Kiểm định profile và nghiên cứu người dùng ở giai đoạn phát triển không phải
là can thiệp sửa mesh thủ công trên từng đơn hàng. Từ chối dữ liệu không đủ là
một kết quả hợp lệ; đồng thời phải báo tỷ lệ từ chối để tránh làm đẹp số liệu.

## Lộ trình và điều kiện chuyển giai đoạn

| Giai đoạn | Trọng tâm | Bằng chứng để chuyển bước |
|---|---|---|
| V2 MVP | Hai mẫu, khớp đầu, màu/biểu cảm, xuất và kiểm tra | Mẫu đầu–tóc thật được in/đo; lỗi được truy vết; phạm vi đầu vào rõ |
| V2 cải thiện | Nhiều mẫu và dáng đầu; xếp hạng/custom tốt hơn | Dữ liệu nhiều người, ảnh giữ lại, so sánh mù và tỷ lệ lỗi thực |
| V3 cá nhân hóa sâu | Khối tóc có hướng hoặc retrieval+deformation tinh hơn | Vượt đối chứng V2 về giống/hài hòa mà không giảm độ tin cậy chế tạo |

Không dùng số lần lặp hay số ứng viên của một người để thay cho số người độc lập.

## Thí nghiệm đề xuất

- Giữ cố định dữ liệu nguồn và kích thước khi so affine/cage.
- Tách dữ liệu phát triển khỏi người và góc ảnh đánh giá độc lập.
- Thu thập đánh giá mù độ giống, hài hòa và cảm xúc; không gán sẵn điểm chuyên gia.
- In mẫu calibration một lần cho mỗi machine–resin–layer–exposure profile.
- Ghi đầu vào bị từ chối, batch lỗi, tỷ lệ geometry pass, số lần sửa, thời gian,
  chi phí và first-pass yield vật lý. Chưa in thì các tỷ lệ vật lý là chưa có dữ liệu.
- Thử bỏ lần lượt cage, sửa có giới hạn hoặc kiểm tra vertex-link để đo đóng góp.

Những kết quả này mới có thể hỗ trợ một đóng góp nghiên cứu. V2 hiện là tích
hợp kỹ thuật và nguyên mẫu; không tuyên bố đã có thuật toán mới hay tối ưu nhất.
