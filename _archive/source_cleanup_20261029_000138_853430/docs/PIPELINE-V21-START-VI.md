# Memento3D V2.1 — sửa giữ nét mặt, tham chiếu da sáng và tóc ngắn ôm đầu

Đây là bản sửa tập trung cho ba vấn đề của V2.0: mặt bị đổi nét do màu vẽ thay thế, da tối, tóc mẫu quá dày/nhô. Không phải một mô hình đã được chứng nhận sẵn sàng chạy máy in.

## Chạy trên máy hiện tại

Giải nén **nội dung ZIP trực tiếp** vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Sau giải nén, `RUN-PIPELINE-V21.ps1` phải nằm ngay trong thư mục đó. Gói có đủ mã phụ thuộc nội bộ và hai STL thư viện; không chứa workspace cá nhân hoặc model parser. Dùng lại workspace và parser có sẵn. Các launcher cũ không bị thay tên.

Từ PowerShell tại thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-PIPELINE-V21.ps1
```

Launcher lấy dự án theo vị trí của chính nó, dùng `.venv\Scripts\python.exe`, tạo `runs\pipeline_v21_<thời gian>` mới và mở `review.html`. Nếu thiếu dependencies CPU, launcher cài vào venv chính. Không cần training hoặc thay môi trường WSL/GPU. Chỉ Bypass cho tiến trình PowerShell này.

Nếu đường dẫn đầu vào khác:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-PIPELINE-V21.ps1 -Workspace "D:\duong-dan\head.workspace.html" -ParserModel "D:\AI-cache\face-parsing\resnet18.onnx"
```

Model mặc định: `D:/AI-cache/face-parsing/resnet18.onnx`. Workspace mặc định được ghi trong `configs/pipeline-v21-04.json`, trỏ đến workspace đã dùng cho mẫu 04. Launcher báo rõ file thiếu trước khi chạy. Chưa thực thi launcher trên Windows trong môi trường kiểm thử này; Python CPU đã chạy với workspace và parser thực tế.

## Thay đổi chính

1. **Mặt:** dùng đúng tọa độ nguồn, chuẩn hóa kích thước giống đầu vào. Mắt giữ geometry và ảnh nguồn. Không pose nụ cười, không vẽ mày/môi kiểu chung. Preview màu dùng texture nguồn; STL không chứa màu. Kiểm tra riêng sai lệch tọa độ khi xuất GLB và các điểm bề mặt mặt phía trước sau boolean.
2. **Da:** tìm vùng semantic skin sáng, liên thông đủ lớn, bỏ pixel cháy sáng. Không lấy pixel sáng nhất đơn lẻ. Không lượng tử về bảng màu V2.0. Điều chỉnh độ sáng có giới hạn bằng trường Gaussian có trọng số mask; giữ sắc độ và tương phản cục bộ. Pixel mắt, mày và môi nguồn được giữ, rồi tile feature từ bản bake nguồn được phục hồi. Màu từ ảnh chưa hiệu chuẩn vẫn chịu ảnh hưởng ánh sáng và camera.
3. **Tóc:** ghép cavity bằng cage trước, sau đó tính giao ray với sọ và thành trong của mẫu. Thành trong được đưa vào sọ 0.12 mm, độ dày thành ngoài được nén. Chạy ba mức 0.15, 0.25, 0.35 trong một batch. Có giới hạn chiều cao đỉnh thêm và kiểm tra khe hở trên các ray mẫu. Không dùng thể tích giao nhau làm bằng chứng duy nhất rằng tóc ôm đầu.
4. **Preview:** texture `KHR_materials_unlit` đã chứa ánh sáng ảnh được render đúng unlit, tránh đánh tối thêm lần nữa.

V2.1 hiện chỉ chấp nhận mẫu tóc ngắn **21778**, chưa áp dụng phép nén cavity này cho tóc dài, búi, bím hoặc bob. Mẫu 11256 vẫn có trong thư viện và V2.0 để đối chiếu, không tự thay cho tóc ngắn của mẫu 04.

## Đọc kết quả

- `review.html`: đầu nguồn, đầu chỉnh sáng, vùng da tham chiếu, ba ứng viên và bản đề xuất.
- `selected/appearance.glb`: màu/texture tham khảo; `selected/head_mm.stl`: khối đầu–tóc đã boolean, đơn vị mm, Z up.
- `selected/hair_mm.stl`: tóc riêng để kiểm tra, **không phải bộ tóc tháo lắp có dung sai thiết kế**.
- `shared/bright_skin.json`: vùng da, RGB và giới hạn điều chỉnh, cờ chưa hiệu chuẩn.
- `*/candidate.json`: số đo fitting, mặt, topology, proxy silhouette. Điểm đề xuất chỉ dùng chọn ứng viên trong batch, không phải thước đo đẹp hay giống độc lập.
- `selected/experimental_supports`: support thử nghiệm, không phải file máy in đã cắt lớp.

STL kín và liên thông chưa đủ chứng minh in được. Chưa có resin/exposure thực tế, kiểm chứng độ dày chính xác toàn bộ, chạy slicer của máy hoặc in vật lý. `ready_for_machine=false` được giữ đúng. Hình học mặt nguồn được giữ không đồng nghĩa nguồn đã tái dựng giống người thật; nếp bóng trong ảnh vẫn có thể xuất hiện ở texture. Đường chân tóc của mẫu thư viện chỉ gần giống, chưa fit chính xác chân tóc cá nhân hoặc sử dụng ảnh sau để dựng gáy. Bản này xử lý khối tóc quá nhô, không tuyên bố giải quyết hoàn toàn kiểu tóc.

## Công thức fitting giới hạn

Với một hướng ray từ bên trong sọ và cavity tóc, gọi `r_s` là bán kính đến sọ, `r_i` đến thành trong tóc, `r` đến vertex mẫu; `t = max(r - r_i, 0)`.

`r_mới = r_s - inset + ratio*t + floor*(1 - exp(-t/0.3))`

`inset=0.12 mm`, `floor=0.18 mm`. Các giá trị là tham số thiết kế thử nghiệm cho đầu/cổ 22 mm, không phải ngưỡng máy in đã hiệu chuẩn. Khi tìm giao ray đầu tiên với tóc, loại các tam giác hướng ra xa điểm xuất phát; tia đi từ ngoài khối tóc vào phải gặp mặt hướng vào trước. Điều này giảm truy vấn tam giác trang trí ngoài nhưng vẫn giữ giao đầu tiên. Truy vấn chia batch 256 và dùng lại correspondence cho ba mức nén. Nếu giao ở mép tam giác bị thiếu, truy vấn lại trên toàn bộ mesh tóc; chỉ cho phép fallback đến vertex nguồn đã biết với tối đa 1% ray. Thiếu nhiều hơn thì loại, không dựng bù. Trên lần chạy mẫu 04, 27 ray thiếu ở mesh lọc đã được truy vấn đầy đủ lại; không ray nào cần endpoint fallback.

Hàm tăng theo `t`, giữ thứ tự thành trong/ngoài theo cùng ray, nhưng không tự bảo đảm toàn cục không tự cắt. Vì vậy luôn kiểm tra volume, manifold sau STL round-trip, contact proxy và vùng mặt bảo vệ. Chỉ dùng khi ray/cavity đáp ứng; thiếu dữ liệu thì loại thay vì tự bù.

## Phạm vi kiểm chứng

Bộ kiểm thử gồm invariants topology/boolean, bảo toàn geometry nguồn, chọn vùng da sáng thay vì bóng/nền/cháy sáng, bảo vệ feature pixels, cavity contact trên mô hình tổng hợp, và render unlit. Kết quả thực tế trên mẫu 04 nằm trong gói Results04 đi kèm, có số đo và ảnh để tự đánh giá. Không dùng số test đạt để suy ra tính thẩm mỹ.
