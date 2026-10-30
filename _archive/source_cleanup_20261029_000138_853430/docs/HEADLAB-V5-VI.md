# HeadLab V5 — tích hợp bốn góc nhìn, kiểm tra và chuẩn bị in thử

## Cài đúng vị trí

Giải nén **Update ZIP** trực tiếp vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Giải nén **Input ZIP** (nhẹ) vào cùng thư mục project để có `headlab-v5-demo/input/back.jpg` đúng ảnh back bạn đã gửi. **Demo ZIP** là tùy chọn lớn hơn, có cùng ảnh đầu vào và toàn bộ kết quả chạy sẵn trong thư mục `headlab-v5-demo/`. Không đổi tên thư mục này nếu muốn dùng lệnh mặc định. Code ZIP không chứa ảnh cá nhân, weights hoặc môi trường Python.

Đây là update trên project đang có, không phải bản cài mới toàn bộ hệ thống tái dựng đầu.

## Một lệnh chạy đầy đủ

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB-V5.ps1
```

Giữ `.venv` CPU hiện tại; không sửa `.venv-rdf`, WSL hoặc conda. Không cài lại thư viện mặc định. Nếu thiếu thư viện, thêm `-InstallDependencies`; dùng cùng stack `requirements-headlab-cpu.txt`, không thêm model weights mới.

Nếu không tải Demo, đặt ảnh back đúng chỗ rồi truyền đường dẫn thật:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEADLAB-V5.ps1 -BackImage "D:\duong-dan-that\back.jpg"
```

Ảnh phải khớp checksum trong study profile case04. Thay người/ảnh cần profile riêng; không tắt checksum để ép dữ liệu khác đi qua.

Mặc định: **2 chiều cao đầu+cổ × 2 nhóm góc nhìn × 5 phương án × 3 lần giải = 60 phép chạy thuật toán**, xuất 20 ứng viên. Mỗi ứng viên chỉ xuất/render một lần. Phần prepare ảnh và bake màu dùng chung.

18/22 mm là hai kích thước thử đầu+cổ, chưa chốt tỷ lệ tượng hoàn chỉnh 80–100 mm. Không có thân hoặc đế trong bản benchmark này.

## Những phần tích hợp thật sự

1. **Ảnh back:** tìm vùng tóc tối trung tâm, tách dựa trên độ sáng và màu đối lập, kiểm tra độ tương phản/diện tích và hai vùng tai. Không remap nhãn 'mũ' thành tóc. Parser mặt có thể nhầm ảnh sau, vì vậy không dùng parser đó làm nguồn duy nhất.
2. **Camera back:** khớp pinhole nhiều khởi tạo với neo vùng tai và đỉnh tóc, có prior góc quay và biến phụ cho độ cao tóc. Đây là khớp gần đúng, không phải calibration được xác nhận. Biến phụ đỉnh tóc chạm giới hạn được ghi rõ.
3. **Tính lại chân tóc:** hợp nhất nhãn nhìn thấy với trọng số theo hướng nhìn; ảnh back có ràng buộc tóc/không tóc. Nền ngoài silhouette ở điểm model nhìn thấy được được dùng làm bằng chứng âm cho tóc, KHÔNG được gọi là da quan sát được.
4. **Điều hòa chân tóc:** regularization trên đồ thị, giữ nhãn chắc chắn; loại vùng tóc hẹp hơn ngưỡng thiết kế thử trên bề mặt tham chiếu. Không ép đối xứng. Các thay đổi footprint được báo riêng, không tuyên bố giữ nguyên mọi quan sát.
5. **Khớp khối tóc:** compact RBF cục bộ với 18 tham số để giảm lan ảnh hưởng từ gáy sang mái; tăng trọng số ảnh trước. So linear/soft_l1 và Laplacian/ARAP cùng dữ liệu, budget, bounds.
6. **Da phía sau:** sửa màu tóc từ ảnh cũ bị bake vào vùng da gáy sau khi chân tóc thay đổi. Chỉ atlas vùng phía sau có bằng chứng màu tóc; không sửa hình học mặt. Màu thay thế là màu da xấp xỉ, không phải albedo sau đầu được đo.
7. **Render/đo nhanh:** vector hóa barycentric rasterizer và z-buffer, kiểm tra đối chiếu với bản tham chiếu. Không dùng ảnh AI làm đẹp kết quả. Framing chung trong mỗi kích thước.
8. **Kiểm tra có tọa độ:** mesh topology và chord screening, xuất điểm bị gắn cờ lên GLB chẩn đoán. Các điểm màu không đi vào STL in.
9. **Tự chọn ứng viên cho thử chuẩn bị in:** lọc hội tụ/topology và giới hạn giảm IoU trước, rồi đề cử theo proxy ba góc + góc sau. Đây KHÔNG phải thuật toán chọn 'đẹp nhất/in được chắc chắn'.
10. **Chuẩn bị support tự động:** tìm hướng đặt, voxel precheck, tạo cột/tip/raft, thử Boolean union, báo vùng không được đỡ hoặc shaft đi xuyên model. Không sửa tay. Đây là support thử nghiệm, chưa là profile resin đã xác nhận.

## Vì sao chạy cả 3v và 4v?

Hai nhóm dùng cùng quy tắc fusion/regularization, cùng người và kích thước. Nhóm 3v không dùng ảnh back để fitting nhưng được đánh giá trên nó. Nhóm 4v sử dụng back nên IoU sau là training consistency, không phải kiểm định độc lập. Camera back cũng được fit từ ảnh này; không coi một con số IoU là độ chính xác 3D.

Không so trực tiếp trung bình ba ảnh với trung bình bốn ảnh. Trang HTML và CSV ghi riêng front/left/right/back. So từng góc và kiểm tra có đánh đổi chính diện hay không.

## Xem gì?

- Demo: mở `headlab-v5-demo/compare.html` trước, chưa cần chạy lại.
- Sau khi chạy: `runs/headlab_v5_<timestamp>/compare.html` tự mở.
- `shared/back.registration.png`: mask xanh, neo ảnh cyan, neo chiếu đỏ.
- `shared/back.observation.json`: miền áp dụng segmentation, giả định camera, sai lệch neo và biến phụ.
- `fusion_3v/fusion.json`, `fusion_4v/fusion.json`: số quan sát, xung đột, phần biên thiếu hỗ trợ và mức giản lược.
- `metrics.csv`: bảng toàn bộ phương án.
- Mỗi ứng viên: `head_mm.stl`, `appearance.glb`, `geometry.glb`, `screening_map.glb`, `chord_samples.json`, `candidate.json` và ảnh overlay bốn góc.
- Thư mục `support_<ứng_viên>/`: đầu đã đặt hướng, supports, assembly chẩn đoán và `support_report.json`. Nếu Boolean union đạt sẽ có `assembly_union_mm.stl`.

STL: mm, Z lên. GLB đầu: mét, Y lên. Support STL giữ cùng hệ tọa độ STL đã đặt hướng. `assembly_unioned_not_verified_mm.stl` là nhiều khối chưa hợp nhất; không nhầm với `assembly_union_mm.stl`.

## Giới hạn quan trọng

- Segmentation back hiện dành cho **tóc tối, nền sáng hơn, thấy hai vùng tai**. Không có model tổng quát cho mọi màu tóc, nơ, mũ hoặc nền phức tạp. Không đủ điều kiện thì dừng, không tự báo thành công.
- Camera được regularize vì các neo gần đúng không xác định đầy đủ pose/focal. Neo khớp tốt không chứng minh camera đúng. Ngưỡng gate chỉ phục vụ thí nghiệm.
- Vẫn là tóc ngắn bám đầu. Tóc dài/buộc/búi chưa được suy ra tự động. Không tuyên bố hỗ trợ chúng chỉ vì có đường cong utility.
- Ngưỡng vùng tóc hẹp `.45 mm` được áp ở đầu+cổ tham chiếu 20 mm; khi đổi tỷ lệ nó thay đổi tương ứng. Đây là giản lược footprint, KHÔNG phải độ dày vách tối thiểu.
- Chord sampling không chứng minh minimum thickness hay sức bền; chưa có kiểm tra giao cắt chính xác hoặc FEA. Topology pass không đồng nghĩa in thành công.
- Support dùng voxel pitch thử `.15 mm`, không phải layer height máy. Có thể bỏ sót đảo nhỏ, shaft chạm model, hoặc để lại dấu khi tháo. Nếu báo unresolved/collision/union fail thì không được diễn giải là support đã đạt.
- Chưa chạy slicer theo định dạng máy Saturn, chưa có resin/exposure/layer height đã hiệu chuẩn, chưa có bản in xác nhận. **ready_for_machine vẫn false**. Không tự động gửi lệnh in.
- Đây là giai đoạn xử lý đầu từ workspace đã dựng; không phải model ảnh thô → đầu mới hoặc tượng toàn thân hoàn chỉnh.

Mục tiêu tự động sản xuất vẫn được giữ: không có bước sửa mesh bằng tay trong batch. Việc đánh giá thí nghiệm/in thử để hiệu chuẩn là công việc phát triển hệ thống, không phải một bước vận hành từng đơn.
