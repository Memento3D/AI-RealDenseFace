# HairBench 1.0.1 — dừng đúng tiến trình và kiểm tra ổ Windows thật

Bản vá không xóa, di chuyển hoặc tạo lại cache/weights/envs hiện có. Mặc định vẫn dùng `/home/indows/.local/share/memento-hairbench-v1`. Giữ nguyên model lock, phiên bản dependencies, input subject và bộ mặt nguồn.

## Sửa gì

- Ctrl+C: gửi SIGTERM tới nhóm subprocess do launcher tạo, đợi tối đa 10 giây rồi SIGKILL nếu chưa thoát; ghi `INTERRUPTED`, giữ báo cáo/đầu ra trước đó và trả mã 130. Không coi hủy thao tác là model inference thành công.
- In nhịp tiến độ mỗi 15 giây trong lúc chạy/tải/cài, kèm kích thước log để cửa sổ không im lặng quá lâu.
- PowerShell tìm ổ Windows chứa WSL qua BasePath trong registry, chuyển root ổ đó sang mount WSL. Controller đọc dung lượng vật lý của mount đó và ổ chứa project, lấy giới hạn thấp nhất cùng giới hạn filesystem Linux; không dùng dung lượng ảo tối đa của VHD làm bằng chứng ổ Windows đủ chỗ.
- Chính sách khởi động: Setup/All tối thiểu 10 GiB trống hiệu dụng; Run tối thiểu 5 GiB. Mỗi nhịp chờ kiểm tra ngưỡng dự phòng 3 GiB, dừng subprocess hiện tại nếu thấp hơn. Đây là ngưỡng bảo thủ của bộ điều phối, không phải chứng nhận dung lượng đủ cho mọi model; có độ trễ kiểm tra 15 giây.
- Run không gọi setup, đồng thời bật HF_HUB_OFFLINE và TRANSFORMERS_OFFLINE để các thư viện này dùng asset có sẵn. Thiếu asset thì báo lỗi, không tự tải qua Hugging Face. Đây không phải sandbox chặn mọi thư viện truy cập mạng.
- Inspect không tải model; báo thêm `existing_setup` để phân biệt chỉ có env với setup đã qua kiểm tra import.

## Cài đúng vị trí hiện tại

Sau khi đã dừng phiên benchmark cũ, giải nén nội dung ZIP trực tiếp vào:

`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Thay mã cùng tên bằng bản mới. Không xóa `runs`, subject, `.venv` hoặc state trong Ubuntu. ZIP không có thư mục state/env/weights và không ghi đè các vùng này. Các file viewer/reference của kit vẫn có đủ trong gói.

## Lưu ý với phiên 1.0 đã bị Ctrl+C

Bản vá không tự truy tìm và giết tiến trình cũ không thuộc lần chạy mới. Nếu còn downloader của HairBench cũ, dừng riêng nó trước khi chạy lại. Không `pkill python` toàn bộ.

```
wsl.exe -d Ubuntu-22.04 -- pkill -TERM -f '^/home/indows/.local/share/memento-hairbench-v1/envs/(triposr|hunyuan-mini)/bin/python -c .*(hf_hub_download|snapshot_download)'
```

Mã trả về 1 của pkill có nghĩa không tìm thấy tiến trình khớp, không phải model bị lỗi.

## Kiểm tra trước, không cài lại

```
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-BENCH.ps1 -Mode Inspect -Distro Ubuntu-22.04
```

Nếu TripoSR đã có kết quả thành công trong report của lần cũ, mở viewer trước; không cần inference lại. Nếu chưa có kết quả nhưng `existing_setup.triposr.setup_checked=true`, có thể thử riêng môi trường đã chuẩn bị:

```
powershell -ExecutionPolicy Bypass -File .\RUN-HAIR-BENCH.ps1 -Mode Run -Models triposr -Distro Ubuntu-22.04 -SubjectDir ".\runs\hair_bench_20261028_173257_386\subject"
```

Nếu Hunyuan chưa có `hunyuan-mini.setup.json`, việc tạo env chưa đồng nghĩa setup/weights hoàn tất. Chưa chạy Setup/All khi D chỉ còn khoảng 8 GiB. Sau khi giải phóng đủ dung lượng và xác nhận không còn downloader cũ, tiếp tục riêng Hunyuan bằng `-Mode All -Models hunyuan-mini` với cùng SubjectDir; bộ cài dùng lại env/cache/local weights đã có. Phần đang tải dở có tiếp tục được hay không phụ thuộc thư viện tải và trạng thái file, không cam kết không có byte nào bị tải lại.

DiffLocks vẫn bị deferred; không tạo một env thứ ba hoặc tải checkpoints của DiffLocks trong bản này.

## Kiểm chứng

14 unittest CPU đạt, gồm giữ checksum mặt, mask crop, trạng thái inference trung thực, xử lý process-group khi hủy, dừng trước launch khi dung lượng thấp, guard ổ vật lý, báo cáo hủy và chế độ HF offline của Run. Python compile đạt.

Không chạy inference GPU hoặc PowerShell/WSL thật trên máy người dùng trong môi trường phát triển. Test không chứng minh VRAM đủ, model tạo tóc đạt chất lượng, hay mesh in được. Không có thao tác dọn cache hoặc compact VHD tự động trong bản vá.
