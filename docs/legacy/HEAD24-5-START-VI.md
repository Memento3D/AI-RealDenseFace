# Head24.5 — màu đa góc, kính 3D, chọn tóc có kiểm tra

## Chạy ngay

Giải nén gói Update trực tiếp vào thư mục gốc project Memento3D-Head-Reconstruction,
ghi đè các file Head24 có cùng tên. Đây là supplement đầy đủ nhánh Head24, gồm
cả các sửa24.1–24.4; cần giữ source V2.3, assets tóc21778/11256 và vendor đang có.
Không xóa input/runs, không thay .venv/.venv-rdf, không tải model AI mới.
Không sửa RUN-HEAD24.ps1 cũ; bản mới dùng lệnh riêng:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD245.ps1 -ResumeRun ".\runs\head24_20261029_044424_683718"
```

Lệnh xác minh crop/fit đã lưu, dựng workspace mới rồi chạy figurine trên CPU.
Không chạy lại detector hoặc RDF/GPU. Kết quả nằm ở run mới head245_resume_*;
không ghi đè các lần thử cũ. Có thể mở review.html trong gói Results05 để xem
kết quả đã chạy sẵn, không phải chạy lại chỉ để xem.

Với bộ dữ liệu mới chưa fit:
```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD245.ps1 -InputDir ".\input\05"
```

## Những thay đổi có chủ đích

### Da

Chỉ dùng semantic skin được erosion, tránh vùng gần tóc/kính/mắt/môi, vùng
bão hòa, quá tối và biên nhiễu. Tính median vùng sáng ổn định78–95% trong mỗi
ảnh; chọn vùng có độ sáng tốt hơn với phạt biến động màu. Không lấy pixel sáng
nhất và không quy màu da theo sắc tộc. So chênh lệch CIEDE2000 giữa các góc để
ghi mức tin cậy. Màu thống nhất ở mặt, tai, cổ; mắt/mày/môi xử lý riêng.

Bộ05 chuyển từ RGB107,100,113 (front ngược sáng) sang vùng right sáng hơn.
Đây là ước lượng màu từ ảnh, chưa phải albedo, white balance chuẩn hoặc màu in.
Chênh lệch ánh sáng lớn vẫn được ghi LOW, không đổi thành kết quả đã hiệu chuẩn.

### Mắt, lông mày và kính

Bỏ photo kính/reflection khỏi texture da; không dùng kính tối làm texture mắt.
Nếu mắt bị kính che và có landmark front khớp SHA pixel, dùng iris trung tính
trên chính mesh nhãn cầu nguồn. Đây là thiết kế ước lượng, không khôi phục mắt,
mí hay gaze bị che. Mày thiếu semantic dùng polygon từ landmark và màu thiết kế;
giữ mày có mask. Môi giữ vùng quan sát có fade/bù màu cục bộ, không thay mesh.

Gọng kính là các tube kín theo bounds lens quan sát được, mặt phẳng chung theo
proxy độ sâu mặt và càng kính hướng về tai. Giữ phom kính đều, không kéo từng
đoạn gọng theo gò má. Gọng được Boolean union vào đầu khi xuất STL; không có
surface tròng kính, khớp tháo lắp hoặc đo quang học. Bán kính tube mặc định0.22mm
(đường kính danh nghĩa0.44mm; tiết diện đa giác có bề dày chiếu nhỏ hơn đôi chút).
Đây là kích thước thiết kế thử nghiệm, chưa phải ngưỡng resin đã hiệu chuẩn.

### Tóc

Thử hai assets có sẵn21778 và11256, mỗi mẫu có body0.60/0.85mm. Radial cavity/scalp
fit vẫn kiểm tiếp xúc, thể tích và bảo vệ mặt; bỏ sàn giữ khối ngoài làm tóc quá
phồng ở V2.3/24.4. Detail mảng lớn tách khỏi độ dày nền, strength0.65. Mẫu21778
có policy tóc mai đã có handles; mẫu11256 giữ boundary mẫu, không áp phép kéo tóc
mai của mẫu khác. Profile cho phép khai báo asset nào có local tip handles.

Kiểm tra tuyệt đối số điểm tai quan sát bị tóc che, ngoài gate so sánh trước/sau
biến dạng. Ngưỡng10% chỉ là guard kỹ thuật thử nghiệm. Chọn điểm silhouette tốt
nhất trong các ứng viên qua geometry/contact/ear guards; không gọi là điểm đẹp,
likeness hoặc độ chính xác tái dựng tóc. Không nâng ngưỡng để ép11256 vào đầu.
Không có ảnh back trong bộ05 nên gáy vẫn là prior từ template.

### Hình học, in và trạng thái

Giữ geometry mặt/mắt nguồn, không làm đẹp hoặc đối xứng hóa hình dạng. Surface
guard mặt được đo sau ghép tóc và trước thêm kính; không được hiểu là toàn bộ
bề mặt STL cuối giữ nguyên vì gọng được cố ý thêm lên mặt.

Union dùng double precision/Manifold và đọc lại STL float32. Chỉ được bỏ fragment
số học nhỏ đồng thời có volume<1e-7mm3, bbox diagonal<0.02mm và tổng<1e-6mm3;
phần rời thật hoặc mảnh dài vẫn bị từ chối. Metadata union ghi việc xử lý nếu có.
Không voxelize mặt. Kính/đầu sau union phải một thành phần và qua mesh audit.

Trang review tách trạng thái xuất file, ngoại hình và sẵn sàng máy in. Hỗ trợ tự
động vẫn là thử nghiệm voxel, không có slicing máy/resin calibration/in thật.
ready_for_machine=false. STL không có màu; GLB màu là preview/painting guide.
Đây vẫn là phần đầu/cổ22mm, chưa phải toàn thân có đế80–100mm.

## Tùy chỉnh có provenance

Sao chép configs/pipeline-v245.json thành file mới, sửa bản sao rồi truyền:
```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD245.ps1 -ResumeRun ".\runs\head24_20261029_044424_683718" -Profile ".\configs\my-head245.json"
```

skin245.override_rgb=null nghĩa là chọn vùng da tự động. Khi đặt RGB cụ thể,
report ghi USER_DESIGN_OVERRIDE. hair245.body_mm, detail_strength, asset_ids và
sideburn_asset_ids cho phép thử có giới hạn. Không sửa config gốc đã có checksum.
GeometryCache là tùy chọn; chỉ nhận checkpoint đúng head/asset/policy/code:
```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD245.ps1 -ResumeRun ".\runs\head24_20261029_044424_683718" -GeometryCache ".\runs\<run-v24-4>\04_figurine"
```

## Tốc độ và kiểm chứng

Môi trường kiểm thử CPU dùng NumPy1.26.4, Trimesh4.7.4, Manifold3D3.1.1,
ONNXRuntime1.20.1, scipy1.15.3, scikit-image0.24.0 và Embreex2.17.7.post6.
Embree là tăng tốc ray CPU tùy chọn, không phải AI/GPU. Máy chưa có vẫn dùng
Trimesh backend hiện tại và có thể chậm hơn nhiều. Không tự đổi môi trường của bạn.
Không có benchmark Windows/RTX4050 mới.

Log test và số liệu chạy thật ở verification/. Giữ nguyên các file runtime V2.3.
Các báo cáo cũ ở HEAD24-START-VI.md chỉ là lịch sử; ưu tiên tài liệu24.5 này.

## Căn cứ phương pháp và giới hạn

- Intrinsic decomposition có bất định từ ảnh đơn: https://rin.csail.mit.edu/
  Bản này không triển khai RIN hoặc tuyên bố giải được albedo; chỉ chọn observation.
- Hair Meshes: https://www.cemyuksel.com/research/hairmeshes/
  Tham khảo tư duy kiểm soát volume; không triển khai strand tracing/simulation.
- Trimesh: https://trimesh.org/ ; Manifold: https://github.com/elalish/manifold
  Geometry validity được kiểm riêng với chất lượng ngoại hình và khả năng in.

Không có tuyên bố nghiên cứu mới hoặc bảo đảm100% in thành công trong bản này.

## Kiểm chứng cuối bộ05

- Hai body21778 qua kiểm tra; chọn0.85mm theo silhouette score. Hai body11256
  bị loại do che16–37% điểm tai; không hạ guard để ép mẫu vào đầu.
- Màu nền177,129,96 từ right. Mức tin cậy LOW do khác ánh sáng giữa các góc.
- Head/glasses STL60.381 đỉnh,120.794 tam giác; kích thước14.494×16.097×22.803mm;
  kín, winding nhất quán, một thành phần, boundary/non-manifold edges0.
- Mắt/mặt GLB so nguồn: max sai số export khoảng8.75e-7mm. Surface guard200 mẫu
  trước thêm kính: max2.37e-7mm. Không phải kiểm chứng likeness hoặc Hausdorff toàn mesh.
- Support mới thử6 orientations và chọn35°/yaw180°: coarse voxel precheck không
  còn unresolved islands, không có shaft collision voxels; union support kín1component.
  Vẫn chưa slice thực tế, đo sức bền hoặc in. Không được suy thành ready_for_machine.
- Geometry/appearance chạy thực tế sau fit được cung cấp, khoảng100.9s ở CPU host
  có Embree. Stage tìm support được chạy bổ sung khoảng14.8s, geometry không đổi.
  Không đo detector/RDF GPU/Windows trong thời gian này.
- 68 test nhánh Head24 qua, architecture PASS. Full suite364 test chưa xanh:
  backend chuẩn còn1 failure/2 errors có từ baseline (thiếu viewer và pyhocon).
  Với Embree, baseline và bản mới đều có2 failures/6 errors; các lỗi tăng thêm
  thuộc test legacy ở các trường hợp tia/surface biên. Không có tên failure mới
  so với baseline dùng cùng backend. Backend có thể ảnh hưởng numerical behavior;
  kiểm tra geometry/contact vẫn được chạy trên đầu ra thật, không chỉ dựa tests.
