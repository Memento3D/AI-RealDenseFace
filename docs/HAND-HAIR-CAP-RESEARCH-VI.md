# Đổi hướng: bàn tay đầy đủ, tóc tùy chỉnh và mũ theo tóc

Ngày đối chiếu nguồn: 2026-11-04. Đây là báo cáo nghiên cứu và chuẩn bị asset; không tuyên bố đã thay được tay hoặc đã thêm tóc vào model hiện tại.

## Kết luận từ model hiện tại

Ảnh cận tay của `assembly-cohesion-01` vẫn cho thấy các đầu ngón bị cụt, dù đã vá bề mặt và đưa tay gần mũ. Các kiểm tra không xuyên hộp mặt mũ, normal và checksum không đủ để kết luận bàn tay đúng hình dáng. Không nên tiếp tục dùng bản vá đó làm template cuối.

Đề xuất thay cả bàn tay tại cổ tay bằng mesh hoàn chỉnh có năm ngón, rồi tạo pose theo điểm tiếp xúc. Giữ áo/sleeve làm vùng che đường nối cổ tay. Đây là chuẩn bị template một lần; không yêu cầu sửa tay cho mỗi người dùng.

## Nguồn đã kiểm tra

### Bàn tay và rig

- [Blender Demo Files — Human Base Meshes](https://www.blender.org/download/demo-files/): bộ v1.4.1 được liệt kê CC0; phù hợp làm nguồn mesh tay để chỉnh template.
- [Thư mục tải chính thức](https://download.blender.org/demo/asset-bundles/human-base-meshes/): đã tải đúng archive v1.4.1, dung lượng 50.643.039 byte. SHA-256 quan sát cục bộ `811f43accbb31a88266d932f8f5563b2d13586fca0ba2693aad1f5fe582b3515`; không có checksum nhà phát hành để đối chiếu độc lập trong lượt này.
- Kiểm tra bằng Blender local: mesh `Hand  - Realistic` có 831 đỉnh / 820 polygon ở base level, tác giả Dan Ulrich, metadata CC0, có Multires; mesh `Hand - Stylized ` có 871 đỉnh / 880 polygon, tác giả Julien Kaspar, metadata CC0. Cả hai chưa có vertex groups/rig trong mesh được nạp. Không gọi chúng là bàn tay đã rig sẵn.
- [Rigify super_finger — mã nguồn Blender](https://github.com/blender/blender-addons/blob/main/rigify/rigs/limbs/super_finger.py): có các điều khiển FK và IK tùy chọn cho ngón, phù hợp để thử pose theo điểm tiếp xúc. Không tự dựng lại hình học ngón đã mất.
- [MPFB trên Blender Extensions](https://extensions.blender.org/add-ons/mpfb/): cung cấp tạo người, rigging, Rigify và IK/FK. Đây là phương án khác để tạo tay/thân nền hoàn chỉnh; cần kiểm thử trên scene của mình.
- [MakeHuman Hands 01](https://www.makehumancommunity.org/assets/assetpacks/hands01.html) là pack deformation targets, không phải một bàn tay mesh rigged độc lập để tải rồi ghép thẳng.

Asset và kết quả kiểm tra nằm trong `runs/hand-hair-research-01/`, không đưa vào Git/code archive. `hand-library-inspection.json` là bằng chứng kiểm tra local, không phải kết quả ghép tay thành công.

Đã xuất hai Blender scenes riêng để bắt đầu làm pose: `hand-realistic.blend` và `hand-stylized.blend`. Chúng giữ nguyên mesh gốc và chưa có rig/chưa lắp vào body. `candidate-manifest.json` lưu tác giả, metadata license và checksum của từng file xuất. Xem nhanh bằng `runs/hand-hair-research-01/review.html`.

### Tóc tùy chỉnh

- [Blender Shrinkwrap Hair Curves](https://docs.blender.org/manual/en/5.1/modeling/geometry_nodes/hair/deformation/shrinkwrap_hair_curves.html) có bám bề mặt, offset và giữ chân tóc. Dùng làm cơ sở cho gắn chân tóc; không tự tạo khối tóc kín để in.
- [Deform Curves on Surface](https://docs.blender.org/manual/id/4.0/modeling/geometry_nodes/curve/operations/deform_curves_on_surface.html) mô tả liên kết chân curve bằng UV và biến đổi của bề mặt. Phải giữ contract UV/correspondence khi đổi đầu.
- [MPFB 2.0.18 release notes](https://www.makehumancommunity.org/mpfb/releases/release_2018.html): bản 2026-10-10 sửa Hair Editor cho Blender 5.2. Chỉ xác minh tài liệu; chưa cài và chạy Hair Editor local trong lượt này.
- [Digital Salon — repo chính thức](https://github.com/digital-salon/Digital-Salon): hướng grooming/mô phỏng tóc và lựa chọn kiểu qua text, có thể chỉnh tương tác. Repo lưu ý bộ dữ liệu công khai khác demo và head asset cần lấy riêng. Không thấy pipeline được tài liệu hóa để tự chuyển ảnh người của mình thành tóc mesh kín cho in resin.
- [Digital Salon INSTALL](https://github.com/digital-salon/Digital-Salon/blob/main/INSTALL.md): có cả hướng Windows và Linux; core cần CUDA/CMake, phần Copilot có env và dữ liệu riêng; AI rendering dùng ControlNet. Không đánh đồng ảnh render AI với geometry tóc 3D đã thay đổi. Chưa benchmark RAM/VRAM trên máy mình, không mặc định bắt buộc WSL hoặc cloud.
- [SRM-Hair — repo tác giả](https://github.com/wang-zidu/SRM-Hair): hướng nghiên cứu hair mesh độc lập đáng theo dõi, nhưng tại thời điểm kiểm tra repo chỉ có README/teaser và ghi coming soon. Chưa có implementation/weights đủ để hứa chạy thử ngay.

### Nghiên cứu sát yêu cầu in

[Chen, Chen, Lu — 3D Printed Hair Modeling from Strand-level Hairstyles, 2022](https://www.sciencedirect.com/science/article/pii/S1524070322000121), DOI `10.1016/j.gmod.2022.101135`, nghiên cứu gom bó sợi, lấy bề mặt bao và dựng mesh bằng Poisson reconstruction để tạo tóc kín từ strand data. Điểm áp dụng cho mình là tách bước tạo/groom tóc khỏi bước tạo geometry để chế tạo. Kết quả và thời gian của bài không phải benchmark của máy hay máy in của mình. Chưa kiểm tra được bản code thực thi của phương pháp trong lượt này.

## Hướng MVP đề xuất

### 1. Thay bàn tay, không kéo dài các đầu ngón bị cụt

Chuẩn bị một mesh tay hoàn chỉnh, có marker cổ tay và rig ngón. Thử pose ngón cái/ngón trỏ chạm mép mũ hoặc nắm dây tua, các ngón khác gập tự nhiên. Pose phải dựa trên từng ngón; dịch cả bàn tay không đủ để đóng khoảng kẹp.

Khóa scale mũ/đầu/tóc trước khi đặt các marker tiếp xúc. Có thể thử tối ưu pose với chi phí gồm khoảng cách đầu ngón đến điểm tiếp xúc, phạt xuyên mesh và phạt lệch pose nghỉ. Đây là công thức thiết kế đề xuất, chưa phải solver đã triển khai. Cần cả kiểm tra hình dáng từ nhiều góc, không chỉ khoảng cách marker.

Nếu nối cổ tay không sạch, dùng cuff áo che vùng nối hoặc chuẩn bị lại template cuff một lần. Sau đó freeze asset pose cho từng kiểu body.

### 2. Thư viện tóc dạng khối có tham số

Ưu tiên ba preset thử nghiệm: tóc ngắn, rẽ ngôi, vuốt lệch. Mỗi preset gồm nền scalp, các lọn chính và vài shape keys như hairline, độ dài mái, độ phồng, hướng rẽ, sideburn và màu. Đây là số lượng/style đề xuất, chưa có ba asset hoàn chỉnh.

Chân tóc gắn bằng UV/barycentric anchors vào scalp đã biết topology. Chỉ warp phần nền/chân tóc theo đầu; giữ silhouette của lọn và không shrinkwrap toàn bộ tóc sát da đầu. Tóc không được kéo biến dạng mắt/mũi/hàm.

Đối với bản in, tạo lọn thành mesh có thể tích và các rãnh đủ lớn sau khi chốt kích thước vật lý. Không coi hair curves, hair cards hoặc vài nghìn sợi mảnh là bản in hoàn chỉnh. Chưa chốt bề dày hay kích thước rãnh vì model hiện chưa được hiệu chuẩn về kích thước in, resin và quy trình hỗ trợ.

### 3. Tạo hai biến thể: có mũ và không mũ

Mũ phải lắp theo envelope **đầu + tóc**, không chỉ theo đầu trọc rồi để tóc xuyên vào. Dùng cùng hairstyle gốc với shape key đội mũ: nén phần tóc bị che và giữ mái/mai/phần gáy nhìn thấy được. Không cắt tóc thành mặt phẳng ở viền mũ.

Tách `Cap_Band`, `Cap_Board`, `Cap_Button`, `Cap_Cord`, `Cap_Tassel`. Thử mép bo nhẹ, vành vải ôm theo scalp/hair và vài nếp vải có kiểm soát để tránh cảm giác trụ cứng/ô úp. Độ nghiêng, kích thước và khoảng hở là tham số cần render so sánh, không mặc định theo một góc hay một số mm được gọi là chuẩn.

Tay cầm và dây tua nhận marker từ mũ cuối; chỉ có một nguồn geometry tua. Khi đổi tóc/mũ, pose tay cần đánh giá lại thay vì dịch một đoạn dây để che sai lệch.

## Thứ tự kiểm thử và tài nguyên

1. Kiểm tra bàn tay CC0 riêng, nhận diện đủ năm ngón, tạo rig/pose và thử một vị trí cổ tay trên body. Render cận tay trước/sau/bên và toàn thân. Đây là ưu tiên đầu tiên.
2. Làm một kiểu tóc ngắn dạng khối cho đầu input04, render không mũ và có mũ. Chưa cần AI sinh tóc, cloud hay bộ dữ liệu grooming lớn.
3. Chỉnh mũ theo đầu/tóc đã chốt; đánh giá lại tiếp xúc từng ngón và dây tua.
4. Kiểm tra topology, giao nhau, kích thước vật lý và slicer trước khi quay lại API; không chỉ đánh giá ảnh đẹp.

Blender base assets/rig/preset được đề xuất cho đường local đầu tiên. Chưa đo hiệu năng pose/groom mới trên máy này. Mỗi thử nghiệm dùng thư mục mới, giữ nguồn/phiên bản/checksum và tránh cài thêm nhiều environment song song.

Nên giữ Digital Salon làm nhánh nghiên cứu sau khi một preset tóc và tay hoàn chỉnh đã chạy ổn. Các asset mẫu được chuẩn bị/chỉnh một lần vẫn phù hợp mục tiêu tự động cho mỗi lần tạo kỷ vật của khách.

## Kiểm tra workspace

Không sửa code runtime/API hoặc model pins trong lượt nghiên cứu này. Architecture PASS. Unittest đã chạy: 562 tests, 1 failure, 4 errors, 1 skipped, cùng các nhóm workspace resume / hair manifest / im2 đã ghi nhận ở lượt trước; log lưu tại `runs/hand-hair-research-01/unittest.log`. Không ghi nhận mọi test đã pass.
