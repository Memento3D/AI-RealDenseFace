# HeadLab V5 — công thức, nguồn và phạm vi bằng chứng

Đọc cùng HEADLAB-V4-METHODS.md. V5 không tuyên bố là thuật toán mới đã được công bố; đây là tích hợp thực nghiệm của nhóm với các thành phần có nguồn gốc rõ ràng.

## 1. Quan sát phía sau

Parser mặt đang dùng có thể nhận tóc sau đầu thành 'hat'. V5 dùng nhánh màu có miền áp dụng hạn chế: tóc tối trên nền sáng, có hai vùng tai. Từ RGB, độ sáng L và opponent c=(R-B)/(R+B+1):

    probability = sigmoid((90-L)/12) * sigmoid((0.08-c)/0.04)

Tìm vùng tối trung tâm, threshold, closing và giữ connected component phù hợp. Kiểm tra diện tích và tương phản trước khi dùng. Đây là heuristic của nhóm, không phải xác suất đã hiệu chuẩn hay một model segmentation tổng quát. Không remap toàn bộ nhãn mũ thành tóc.

## 2. Camera phía sau

Camera pinhole p~K[R|t] với focal giả định, góc quanh hướng rear, log-scale và principal point được fitting bằng bounded robust least squares nhiều khởi tạo. Hai vùng tai là neo gần đúng; đỉnh tóc dùng thêm một biến nuisance độ cao 0–3 mm, không phải điểm đầu sọ đã biết. Pose priors regularize yaw/pitch/roll vì dữ liệu không đủ định danh đầy đủ camera.

Báo cáo ghi focal chưa calibrated, prior yaw và trường hợp nuisance chạm bounds. Ngưỡng lỗi neo <=8 px là gate thí nghiệm. Sai lệch neo thấp KHÔNG phải độ chính xác độc lập của camera hoặc hình 3D. Không dùng ảnh back để texture lại mặt phía trước.

Nguồn solver: https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html

## 3. Hợp nhất đa góc và tính lại chân tóc

Ở mỗi vertex, trọng số theo hướng nhìn là cos^4, sau kiểm tra z-buffer/occlusion. Quan sát back được nhân trọng số cấu hình. Trung bình có trọng số cho scalar nhãn tóc. Ghi xung đột giữa các góc; vùng khuất dùng harmonic interpolation. Không ép đối xứng.

Đặc biệt, ở điểm bề mặt model nhìn thấy từ camera back đã được chấp nhận, pixel ngoài hair silhouette được dùng làm bằng chứng âm cho tóc. Điều này không có nghĩa pixel nền là da. Vì camera gần đúng, ràng buộc này phải đi kèm kiểm tra sai lệch và giới hạn miền áp dụng.

Điều hòa scalar trên đồ thị:

    minimize sum_i c_i (f_i-y_i)^2 + epsilon sum_i (f_i-f_initial_i)^2
             + lambda sum_edges (f_i-f_j)^2

Các nhãn quan sát chắc chắn không xung đột và closure nhân tạo làm neo. Sau đó trích iso-contour f=0.5. Chân tóc được tính lại trước bước dựng khối; không tiếp tục khóa đường gáy V4 cũ.

## 4. Giản lược phần tóc quá hẹp

Geodesic opening trên đồ thị có trọng số độ dài cạnh: lấy lõi cách vùng không tóc >=r, rồi mở rộng lại <=r. Những nhánh hẹp không có lõi bị loại. Ngưỡng trial width 2r=.45 mm áp trên đầu+cổ tham chiếu 20 mm, scale theo ứng viên. Đây là quy tắc giản lược footprint của tóc, KHÔNG phải chứng nhận wall thickness hoặc ngưỡng vật liệu Saturn. Số vertex bị giản lược được ghi rõ.

API shortest-path: https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.sparse.csgraph.dijkstra.html

## 5. Compact RBF và fitting khối tóc

Thay Gaussian hỗ trợ toàn cục bằng kernel compact kiểu Wendland:

    phi(r) = max(1-r,0)^4 * (4r+1)
    B_ij = phi(distance(p_i, center_j)/radius) / sum_k phi(...)

18 tâm được chọn theo farthest-point sampling, radius phủ hết điểm với ít chồng lấn hơn Gaussian. Đây là partition-of-unity parameterization của nhóm dùng kernel tham khảo; không tuyên bố bảo đảm tối ưu toàn cục hoặc là nguyên bản ứng dụng tóc của bài báo.

Nguồn kernel: Holger Wendland (1995), Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree.
https://doi.org/10.1007/BF02123482
https://eref.uni-bayreuth.de/id/eprint/55365/

Giữ hai loss linear/soft_l1 và hai regularizer uniform Laplacian/ARAP như V4. Các view weights được ghi rõ. RBF bounds và số controls chạm bounds được báo; không coi hội tụ là xác nhận kiểu tóc đúng. Không có hướng sợi được tái dựng ở đây.

## 6. Rasterizer nhanh

Mở rộng barycentric rasterization/z-buffer tham chiếu bằng NumPy chunking: duyệt cùng pixel nguyên trong bbox tam giác; độ sâu projective là 1/sum(bary_i/z_i). Shading/UV trong preview orthographic dùng cùng phép nội suy với reference. Có tests đối chiếu depth và ảnh render. Đây là tối ưu triển khai, không phải ảnh generative hoặc chứng minh thời gian thực trên mọi máy.

## 7. Chẩn đoán và support

Chord ray screening được bổ sung area-stratified sampling và tọa độ điểm; không chứng minh minimum thickness. Điểm màu là marker riêng, không thêm vào STL in. Xem nguồn Moller–Trumbore và giới hạn ở tài liệu V4.

Tìm hướng đặt dùng cùng sáu phương án nhưng yaw TRƯỚC tilt để thay đổi hướng trọng lực tương đối. Score diện tích mặt hướng xuống có trọng số bảo vệ mặt trước vẫn chỉ là proxy.

Voxel precheck dùng occupancy solid ở pitch .15 mm. Connected components từng lớp được so với lớp trước; các thành phần mới được chọn điểm neo trong giới hạn budget và vùng bảo vệ mặt. Tự tạo shaft, tip frustum và raft, kiểm tra shaft/model collision trên voxel rồi thử Boolean union. Xuất riêng cả dữ liệu thất bại. Pitch voxel này KHÔNG là layer height máy, không thay thế kiểm tra lớp từ slicer.

## 8. Chọn ứng viên cho bước chuẩn bị in

Lọc topology/hội tụ, giới hạn giảm IoU trước so với baseline cùng group/kích thước. Sau đó đề cử bằng score proxy ba view + back. Đề cử chỉ quyết định thử support nào; không phải thuật toán xác nhận nhận diện hoặc khả năng in. 3v/back là held-out với fitting tóc nhưng camera back vẫn fit từ ảnh; 4v/back là training consistency. Không đánh đồng hai dạng bằng chứng.

## 9. Ranh giới của bản update

- Đầu từ workspace đã dựng, không phải model tái dựng đầu mới từ ảnh thô.
- Tóc ngắn bám đầu; chưa có inference tổng quát tóc dài/buộc/búi/phụ kiện.
- Mặt/biểu cảm gốc được giữ để không làm thay đổi nhận diện ngoài phạm vi thử. Màu mặt và mắt không phải đầu ra màu máy resin.
- Chưa có FEA, kiểm tra chính xác minimum-wall/self-intersection hoặc profile resin đã hiệu chuẩn.
- Chưa xuất/chạy file slicing theo định dạng máy Saturn. File support union có topology tốt vẫn không được gọi là machine-ready.
- Không có bằng chứng in vật lý. Tất cả ready_for_machine vẫn false.
