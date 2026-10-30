# HeadLab V4 — phương pháp, công thức và căn cứ

Tài liệu này mô tả đúng phần code có trong bản update. Đây là một benchmark nghiên cứu, chưa là quy trình sản xuất đã được xác nhận.

## 1. Các bước dùng chung

Đọc đầu 3D và ba ảnh trong workspace. Giữ model đầu, camera và các mask giống nhau giữa các ứng viên. Mô hình BiSeNet ONNX được pin SHA256 theo project. Gán nhãn tóc/da trên bề mặt theo độ nhìn thấy; vùng không quan sát dùng nội suy harmonic có điều kiện biên. Đóng biên cổ/miệng nhân tạo không được sinh tóc. Không có camera ảnh sau đầu, không suy ra chân tóc phía sau là dữ liệu đã quan sát.

Phạm vi hiện tại chỉ gồm tóc ngắn bám đầu. Mặt/biểu cảm không đổi để cô lập ảnh hưởng của hai yếu tố loss và regularization. Không đánh đồng việc giữ nguyên model mặt với xác nhận độ giống người.

## 2. Khối tóc ít tham số — thiết kế của nhóm

Với điểm bề mặt p_i, pháp tuyến n_i và taper t_i từ khoảng cách geodesic đến biên:

    h_i = seam + t_i (sum_j B_ij theta_j - seam)
    q_i_target = p_i + n_i h_i

B_ij là trọng số Gaussian RBF đã chuẩn hóa; tâm được chọn bằng farthest-point sampling từ phần tóc có hỗ trợ quan sát và đủ xa biên. theta_j là độ nâng tính bằng mm. Đây là cách tham số hóa thực nghiệm của nhóm, không phải Hair Meshes hoặc tái dựng hướng sợi tóc.

Tối ưu bằng SciPy bounded TRF. Vector residual gồm signed-distance tới mask ở các điểm gần tiếp tuyến nhìn thấy được (đơn vị pixel / pixel_scale), cộng prior (theta_j - base) / prior_mm. Bounds, dữ liệu lấy mẫu, điểm đầu và budget dùng chung. So sánh loss='linear' với loss='soft_l1', f_scale=1. Robust loss áp cho CẢ vector residual, không chỉ pixel residual. Không có ràng buộc phi tuyến chế tạo chính xác trong bước này.

Nguồn API: https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.least_squares.html

Kết quả hội tụ không bảo đảm tối ưu toàn cục. Nhiều theta chạm bounds là dấu hiệu cần kiểm tra mô hình/giới hạn, không phải tự tăng bounds để cải thiện điểm. IoU được đo bằng rasterization sau fitting, không phải objective trực tiếp.

## 3. Uniform Laplacian displacement regularization

Gọi P là bề mặt tóc nền, T là mục tiêu sau fitting, Q là kết quả và L là Laplacian của đồ thị cạnh với trọng số đồng đều:

    E_L(Q) = ||L(Q-P)||_F^2 + lambda ||Q-T||_F^2

Biên chân tóc được cố định bằng điều kiện Dirichlet; giải hệ tuyến tính thưa cho ba tọa độ. Đây là một regularizer dựa trên Laplacian, KHÔNG phải toàn bộ thuật toán LSE gốc với tọa độ bất biến quay.

Nguồn nguyên tắc: Sorkine và cộng sự, Laplacian Surface Editing, SGP 2004.
https://igl.ethz.ch/projects/Laplacian-mesh-processing/Laplacian-mesh-editing/

## 4. Uniform spokes ARAP

Với cạnh e_ij = p_i-p_j, d_ij = q_i-q_j và ma trận quay R_i:

    E_A = 1/2 sum_edges (||d_ij-R_i e_ij||^2 + ||d_ij-R_j e_ij||^2)
          + lambda ||Q-T||_F^2

Local: SVD của covariance từng đỉnh, loại reflection bằng hiệu chỉnh determinant. Global: giải hệ sparse (L + lambda I) cùng neo biên cố định. Lặp số vòng đã chốt trong profile; lưu objective theo vòng. Bản này dùng trọng số đồng đều dương, không tuyên bố là cotangent discretization hoặc code của tác giả.

Nguồn: Olga Sorkine-Hornung, Marc Alexa. As-Rigid-As-Possible Surface Modeling, SGP 2007.
https://igl.ethz.ch/projects/ARAP/

Unit tests kiểm tra rigid rotation, neo biên và objective không tăng trên trường hợp tổng hợp. Những test này không chứng minh bảo toàn nhận diện trên mọi người.

## 5. Kiểm tra hình học

Mặt ngoài tóc được ghép với phần da bổ sung, rồi union cùng mắt bằng Manifold. Mặt gốc không voxel hóa. Kiểm tra watertight, winding, finite, thể tích dương và số thành phần. Boolean có thể thay đổi các vùng giao mắt/tóc trong STL. Kiểm tra này không đủ chứng minh không còn giao cắt, độ dày tối thiểu hoặc in được.

Bản in solid; hair ribbon riêng chỉ là trung gian. Không thiết kế tóc lắp rời. Coplanar simplification dùng dung sai số học 0.0001 mm, không phải độ chính xác máy in.

## 6. Ray screening

Từ các tâm tam giác lấy mẫu tất định, bắn ray theo hướng pháp tuyến vào trong; tìm giao tiếp theo bằng phép ray-triangle Moller–Trumbore. Báo độ dài chord nhỏ nhất đã lấy mẫu và số chord dưới ngưỡng thử. Đó KHÔNG phải minimum wall thickness, CGAL Shape Diameter Function hoặc mô phỏng sức bền. Sampling có thể bỏ sót phần mỏng.

Moller, Trumbore. Fast, Minimum Storage Ray-Triangle Intersection. Journal of Graphics Tools 2(1), 1997.
https://doi.org/10.1080/10867651.1997.10487468

Nguồn đối chiếu cho khái niệm ước lượng độ dày, CHƯA triển khai CGAL:
https://doc.cgal.org/latest/Surface_mesh_segmentation/index.html

## 7. Hướng in và kiểm tra file máy

Thử cùng sáu hướng cho mọi ứng viên, chấm proxy diện tích mặt hướng xuống có trọng số ưu tiên bảo vệ mặt trước. Không dùng proxy này làm bằng chứng support hoặc độ bền. Chưa tự sinh support trong benchmark mặc định.

Adapter bên ngoài có hợp đồng đầu vào/profile/file máy bằng SHA256 và báo cáo kiểm tra. Mặc định không cấu hình adapter, profile resin chưa hiệu chuẩn và kiểm tra độ dày/giao cắt chính xác chưa có. Vì vậy ready_for_machine luôn false. Không chạy UVtools rồi giả định toàn bộ khả năng in được bảo đảm.

Nguồn tham khảo lớp kiểm tra sau slicing: https://github.com/sn4k3/UVtools

## 8. Những nguồn chỉ tham khảo

- Hair Meshes (Yuksel, Schaefer, Keyser, 2009): điều khiển thể tích tóc; không triển khai ở đây. Trang tác giả nêu điều kiện sử dụng/liên hệ.
  https://www.cemyuksel.com/research/hairmeshes/
- Direct shape optimization for strengthening 3D printable objects (Zhou và cộng sự, 2016): tối ưu với mô phỏng và tải xác định; không có FEA trong update.
  https://kalo-ai.github.io/papers/directshapeoptimization/index.html
- Centripetal Catmull–Rom: utility và test đã có, chưa nằm trong năm pipeline. Bảo đảm cục bộ của curve không áp cho toàn bộ bề mặt tóc.
  https://www.cemyuksel.com/research/catmullrom_param/

## 9. Thiết kế thí nghiệm

Baseline + factorial 2 loss × 2 regularizer; hai chiều cao đầu+cổ là các study hypotheses, cùng reference geometry. Mặc định ba lần giải tất định; chỉ để đo thời gian, không phải ba chủ thể độc lập. Cùng framing trong mỗi chiều cao, không có hậu kỳ làm đẹp riêng cho ứng viên. Không đo hiệu năng Windows/GPU từ kết quả CPU môi trường thử.

Các metric fitting-photo chỉ kiểm tra tính nhất quán với đầu vào; thêm người khác, ảnh giữ lại, đánh giá nhận diện và in vật lý là những bằng chứng cần bổ sung. Không tự chọn phương án thắng bằng một metric duy nhất.
