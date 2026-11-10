# HeadLab26.1: phương pháp, đối chứng và giới hạn bằng chứng

## Câu hỏi nghiên cứu có thể kiểm tra

Với cùng ảnh/crop/prediction và budget, giảm ảnh hưởng dense correspondence bị kính/tóc/nền che có cải thiện đường mí quan sát được và độ ổn định mặt hay không? Với cùng mặt, ARAP có bảo toàn phom tóc trong giới hạn tiếp xúc tốt hơn vỏ radial hay không? Màu thống nhất hiện tại và albedo estimate TRUST khác nhau ra sao giữa các điều kiện chụp?

Đây là kết hợp ứng dụng các phương pháp đã có để thử nghiệm. Chưa có cơ sở tuyên bố thuật toán mới hoặc “hiện thế giới chưa có giải pháp”.

## Tài liệu và mã nguồn gốc

1. RealDenseFace: <https://github.com/LinzhouLi/RealDenseFace>. Dùng SDK hiện có và manifest `head24-rdf-source.json`; không thay CUDA solver. Identity chung, expression/pose/FOV riêng từng ảnh. ViT-B được ghi hash cục bộ, chưa có pin checkpoint độc lập.
2. Pixel3DMM — Giebenhain et al., 2025: [paper](https://arxiv.org/abs/2505.00615), [official code](https://github.com/SimonGiebenhain/pixel3dmm), [project](https://simongiebenhain.github.io/pixel3dmm/). Dùng multi-image/discontinuous tracking; không giả định ba ảnh chụp rời có cùng crop-camera. Adapter trong `external.py`, chuyển camera trong `pixel_export.py`; camera cần overlay kiểm chứng trên GPU run thật.
3. TRUST — Feng et al., ECCV2022: [paper](https://arxiv.org/abs/2205.03962), [official code](https://github.com/HavenFeng/TRUST), [project](https://trust.is.tue.mpg.de/). Face và scene dùng để ước lượng albedo/light. Code mới không gọi một median RGB từ toàn UV là skin ground truth.
4. Sorkine & Alexa, *As-Rigid-As-Possible Surface Modeling*, SGP2007: [paper/project](https://igl.ethz.ch/projects/ARAP/), [DOI](https://doi.org/10.2312/SGP/SGP07/109-116), [libigl tutorial](https://libigl.github.io/tutorial/). Bản này tự triển khai local/global ARAP uniform-edge bằng SciPy, **không gọi libigl và không tuyên bố bảo toàn thể tích**.
5. SciPy sparse solve: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.factorized.html>. Huber loss/robust least squares: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>. Bản này dùng outer IRLS với Gauss–Newton solver có sẵn, không gọi SciPy least_squares cho RDF.
6. Trimesh: <https://trimesh.org/trimesh.html>. `is_watertight`, `is_volume` là kiểm tra hình học, không chứng nhận máy in/resin/support.
7. UVtools: <https://github.com/sn4k3/UVtools>. Hook cấu hình theo bản cài thực tế; không suy diễn “không có issues” chỉ từ process exit 0.

## Ràng buộc fit và chính sách trọng số

Với log variance gốc `s`, residual đã whitening có hệ số `exp(-s/2)`. Muốn nhân amplitude bởi `c` thì dùng `s' = s - 2 log(c)`. `c` ở đây là **amplitude**, không phải xác suất hoặc độ tin cậy đã hiệu chuẩn. Config kính/tóc/nền là giả thuyết thiết kế cần ablation.

Huber IRLS: `w = min(1, delta / ||r||)`. Vì `w` là trọng số objective, worker nhân amplitude bằng `sqrt(max(w, minimum_weight))`. Điểm bị che không được eye boost trở lại. Mỗi outer refit dùng cùng identity prior gốc, tránh làm trôi tâm regularization.

Baseline và occlusion dùng cùng ảnh, prediction cache, iteration budget. Tiết kiệm inference neural lần hai; optimizer vẫn fit hai lần. Các điểm source và ảnh được bind bằng hash; không cache chéo người.

## Chẩn đoán và chọn nhánh

- Geometry xám giữ nguyên mesh nguồn, không dùng repaint để che lỗi fit.
- Mí FLAME2023 và contour MediaPipe được so trong pixel camera của từng ảnh. Trung bình khoảng cách gần nhất hai chiều chia khoảng cách hai mắt là **proxy**, không phải Hausdorff chính xác, không phải ground truth anatomy.
- Chiều cao band mí/chiều rộng là proxy khác với aperture vật lý thật. Detector bị kính che cũng có thể sai.
- Khi front có kính/không có semantic eye support, không tự chọn model “tốt hơn” nhờ proxy mắt thấp. Giữ baseline trước; xuất cả hai để kiểm tra.
- Sửa mắt nguồn cần bằng chứng từ fit mới. Nhánh màu, kính và tóc không được ghi là đã sửa anatomy mắt.
- Màu da đa góc là heuristic trên vùng semantic skin đủ sáng, trim outliers; không lấy pixel sáng nhất/chói nhất làm màu thật. Chi tiết quan sát của môi/mày được local Lab rebase, không hứa tách hoàn toàn ánh sáng khỏi sắc tố.

## ARAP tóc

Năng lượng: tổng bình phương chênh lệch cạnh biến dạng với cạnh gốc xoay bởi rotation cục bộ; rotation tính SVD, vị trí free vertices giải sparse với hard handles. Sau giải kiểm finite, displacement budget, positive closed volume. Kiểm tiếp xúc, vùng mặt được bảo vệ, tai, crown và STL cuối vẫn thực hiện riêng.

**Bản 26.0:** asset21778 bị từ chối vì >60k đỉnh; asset11256 vượt movement budget3mm. **Bản 26.1:** cluster vị trí theo lớp trong/viền/ngoài tạo control graph lấy cạnh từ topology gốc. Giải local/global ARAP trên graph, truyền 4-nearest inverse-distance displacement trong từng lớp, khóa vùng viền/crease về radial reference, làm mượt residual24bước và line-search chống đảo hướng tam giác. Các bước clustering/transfer/limiter là quyết định triển khai thực nghiệm của project, không phải kết quả được paper ARAP bảo đảm. Không gọi đây là cage được chứng nhận hoặc injective deformation.

Replay05: 88.350 đỉnh thành 4.181 control nodes; hai biến thể21778 xuất được sau đầy đủ gate. Hybrid giữ 12.5% residual sau filtering; khác biệt radial nhỏ. Mẫu 11256 vẫn bị loại. Không suy diễn từ topology pass sang độ giống/thẩm mỹ/in thành công.

## Bảng kiểm release

Xem `verification/head261-release.json`, log test mới và full-suite baseline/after. Các bước đã chạy CPU có số liệu thật. Chưa chạy mới RDF CUDA, Pixel3DMM hoặc TRUST ở máy phát hành; chưa kiểm Windows PowerShell runtime; chưa in vật lý. Bản release là công cụ chạy thử và so sánh có kiểm soát, chưa phải chứng minh phục hồi khuôn mặt hoặc tự động in thành công.
