# Memento3D Hair Studio V2.3

Đổi kiểu tóc trực tiếp trong trang HTML, từ folder `assets/hair`, giữ nguyên model mặt và màu da V2.2. Không cần chạy lại AI/training mỗi lần thử mẫu tóc. Có sẵn hai STL đã dùng trong dự án.

## 1. Cài đúng thư mục hiện tại

Giải nén **nội dung gói mã trực tiếp** vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Sau giải nén phải có `RUN-HAIR-STUDIO.ps1`, `scripts/hair_studio.py` và `assets/hair-studio/index.html` ngay trong cấu trúc dự án hiện tại. Gói gồm đầy đủ code V2.2 và phần Studio mới, không chỉ patch vài file. Không ghi đè dữ liệu `runs` của bạn.

Nếu đã có lần chạy V2.2 hợp lệ, chạy tại thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-STUDIO.ps1
```

Launcher tự chọn lần chạy V2.2 hợp lệ mới nhất trong `runs`, rồi mở trang trên `http://127.0.0.1:<port>/hair-studio.html`. **Giữ cửa sổ PowerShell đó mở** khi dùng trang; Ctrl+C để dừng. Nếu cổng 8765 đang bận, chương trình chọn cổng trống khác và mở đúng địa chỉ.

Chương trình dùng `.venv\Scripts\python.exe` và bộ dependencies CPU đã cài cho V2.2. Nếu thiếu, launcher cài theo `requirements-pipeline-v2.txt`; không thay môi trường WSL/CUDA/RDF. Không cần cài Node/npm để sử dụng giao diện. Three.js và các addon đã đóng gói, không nạp từ CDN.

### Chỉ định một kết quả cụ thể

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-STUDIO.ps1 -RunDir ".\runs\pipeline_v22_THOI_GIAN_DA_CHAY"
```

Thay đường dẫn ví dụ bằng folder có thật, chứa `shared/canonical_reference.glb`, `profile.used.json`, `report.json`, `selected/candidate.json` và `selected/hair_mm.stl`.

### Chưa có kết quả V2.2 trên máy?

Giải nén gói demo riêng vào cùng thư mục dự án để có `hair-studio-demo-04/`, rồi chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-STUDIO.ps1 -RunDir ".\hair-studio-demo-04"
```

Demo là dữ liệu đầu mẫu 04 hiện tại, không phải một model khác được tạo minh họa. Gói mã không chứa model đầu cá nhân; model đó nằm trong gói demo riêng.

Nếu cần dựng từ workspace trước rồi mới mở Studio, dùng:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-PIPELINE-V23.ps1
```

Lệnh này chạy V2.2 với cùng các tùy chọn `-Workspace`, `-ParserModel`, `-ReferenceRun`, `-InstallDependencies`, sau đó mở Studio. Không cần chạy lệnh dựng lại này chỉ để thêm hoặc đổi mẫu tóc.

## 2. Thêm tóc: chỉ chép file vào folder

Ví dụ:

```text
assets/hair/toc_ngan_moi.stl
assets/hair/toc_dai_moi.glb
```

Hỗ trợ **STL và GLB**, gồm cả đuôi viết hoa. Có thể đặt trong folder con của `assets/hair`. Trang tự quét khoảng 2,5 giây/lần; nếu đang chép file, mẫu tạm khóa cho đến khi file ổn định. Không cần sửa HTML hoặc thêm thủ công vào `library.json`.

- Bấm tên tóc để thay trên đầu 3D.
- Mẫu chọn trước đó không được phép ghi đè mẫu chọn mới hơn khi xử lý xong chậm.
- Thay nội dung file: trang phát hiện revision mới và nạp lại mẫu đang chọn.
- Xóa file đang chọn: trang bỏ tóc đó, giữ nguyên đầu.
- Chuyển tóc không tải lại đầu, không đổi camera hoặc vật liệu da.
- Mẫu mới khớp sơ bộ theo khung đầu. Có thể chỉnh kích thước, cao/thấp và xoay ngay trên trang.
- Các mức chỉnh được giữ trong phiên cho từng mẫu; bấm Đặt lại để trở về thông số đầu.

**HTML mở bằng file:// không được quyền tự quét folder trên máy.** Vì vậy cần mở bằng launcher/server nội bộ. Nếu bấm trực tiếp file `hair-studio.html`, trang hiện hướng dẫn chạy launcher thay vì một trang trống. Không cần tài khoản hoặc server Internet.

## 3. Mức độ khớp đầu của mẫu tóc

| Loại | Cách xem trước |
|---|---|
| Tóc 21778 trùng asset của kết quả V2.2 | Dùng đúng mesh tóc đã xuất, gồm tóc mai V2.2 |
| Mẫu có SHA khớp `assets/hair/library.json` | Ghép theo metadata cavity của mẫu |
| STL/GLB mới chưa có metadata | Căn khung sơ bộ, giữ tỉ lệ và cho chỉnh trực tiếp |

Không có thuật toán đáng tin cậy nào có thể mặc định mọi STL bất kỳ đều là tóc, biết đúng hướng trước/sau, vị trí cavity và tự trở thành mẫu in hợp lệ. Vì vậy mẫu mới vẫn hiện để thử ngay, nhưng giao diện ghi rõ mức fit. Mẫu GLB nên là **tóc riêng**, không chứa cả đầu hoặc cơ thể; các vật thể trong scene GLB được gộp làm một tóc xem trước.

Mặc định STL được hiểu là **Z up**, GLB là **Y up**. Nếu asset có hệ trục khác, tùy chọn thêm file cùng tên đuôi `.hair.json`, ví dụ:

```json
{
  "name": "Tóc mái lệch",
  "up_axis": "Y",
  "transform": {"scale": 1.0, "height_mm": 0.0, "yaw_deg": 180.0}
}
```

Lưu cạnh `toc_mai_lech.stl` thành `toc_mai_lech.hair.json`. Sidecar không bắt buộc để mẫu xuất hiện. Nó dùng chuẩn hóa hướng/cỡ một lần cho asset, không làm thay đổi model mặt hoặc dữ liệu nguồn.

Mỗi file tối đa 128 MiB, mỗi mesh preview tối đa 500.000 tam giác. Asset hỏng hoặc metadata sai được báo riêng; các mẫu khác vẫn dùng được. GLB dùng buffer nhúng; không hỗ trợ glTF phụ thuộc file buffer ngoài hoặc các định dạng FBX/OBJ trong bản này.

## 4. Điều khiển và xuất

- Kéo chuột xoay, cuộn zoom, chuột phải di chuyển; có nút Trước/Nghiêng/Sau/Tự xoay.
- Vật liệu / Màu nền không đèn / Mesh xám để đánh giá màu và khối.
- **Lưu lựa chọn JSON**: tên file/SHA/revision, nguồn đầu, scale/height/yaw và cờ preview.
- **Tải GLB xem trước**: ghép đầu nguồn và tóc đang chọn theo thông số chỉnh. Buffer vị trí đầu giữ nguyên, chỉ thêm tóc. Xuất vật liệu gốc, không bake chế độ Mesh xám của giao diện.
- Link **Báo cáo hình học** mở báo cáo của lần dựng đầu gốc. Các số đo trong đó không tự áp dụng cho tóc mới hoặc thông số mới chỉnh trong Studio.

Đây là chức năng **chọn và xem tóc**, không phải kiểm chứng in cho mọi asset nhập mới. GLB được đánh dấu `preview_only=true`, `ready_for_machine=false`; tóc chưa boolean với đầu, chưa kiểm tra độ dày/support/mã máy. Không có nút xuất STL giả vờ đã qua kiểm định. Việc chọn mẫu đẹp và kiểm tra in là hai bước riêng trong triển khai hiện tại.

## 5. Phạm vi đã kiểm thử

Có kiểm thử Python cho thêm/xóa/sửa asset, partial copy, metadata lỗi, đường dẫn, export giữ buffer đầu; kiểm thử JavaScript cho race chọn mẫu; kiểm thử trình duyệt thật có WebGL với mẫu 04 cho đổi tóc, tự phát hiện file mới, thay/xóa file, xuất GLB và giao diện 390px. Chi tiết trong `HAIR-STUDIO-VALIDATION.md` của gói.

Launcher PowerShell chưa được thực thi trên Windows tại môi trường kiểm thử này. Phần Python/HTTP/HTML/WebGL đã chạy trên CPU/Linux; không phải chứng nhận hoạt động trên mọi card đồ họa hoặc browser Windows.
