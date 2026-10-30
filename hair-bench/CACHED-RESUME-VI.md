# Hoàn tất TripoSR từ cache đã tải — không chạy lại Setup/All

Dành cho trường hợp config.yaml/model.ckpt đã tải thành công vào:
`/home/indows/.local/share/memento-hairbench-v1/cache/huggingface/hub/models--stabilityai--TripoSR`
nhưng lần Setup trước bị ngắt nên chưa hoàn tất local weights/DINO config/import check/setup record.

## Chạy

Giải nén nội dung ZIP trực tiếp vào thư mục dự án hiện tại:
`D:\FPTU\07\EXE201\ai-3d-generation\Memento3D-Head-Reconstruction`

Gói giữ mã Hotfix 1.0.1 và bổ sung hai file hỗ trợ. Không chứa env, cache hoặc weights; không xóa chúng. Chấp nhận thay các file mã cùng tên bằng bản trong gói, giữ nguyên runs và dữ liệu.

Từ PowerShell ở thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\RESUME-TRIPOSR-CACHED.ps1
```

Có thể thêm `-SubjectDir ".\runs\hair_bench_20261028_173257_386\subject"` nếu muốn dùng đúng một subject đã chuẩn bị. Không truyền thì tìm subject gần nhất trong các thư mục hair_bench_*; nếu chưa có, launcher gốc chuẩn bị từ workspace mặc định như trước. Distro mặc định Ubuntu-22.04; script nhắm đúng state/env của user indows đã xuất hiện trong log.

## Thao tác thực tế

1. Kiểm tra repo TripoSR đúng commit trong models.lock.json và không có thay đổi cục bộ. Không checkout/ghi đè repo.
2. Lấy config và checkpoint **chỉ từ cache**, đúng revision đã khóa. Không tải lại checkpoint. Nếu cache thiếu thì báo lỗi.
3. Kiểm tra fingerprint nguồn; nếu tên blob LFS là SHA256 thì so sánh SHA256 nội dung. Tạo symlink từ `weights/triposr` tới cache, không sao chép checkpoint 1.68 GB. File có sẵn giống nội dung được giữ; file khác hoặc symlink hỏng bị từ chối, không bị xóa/ghi đè.
4. Dùng DINO `config.json` từ cache, hoặc tải đúng file JSON nhỏ tại commit đã khóa nếu thiếu. Không tải DINO weights. Download giới hạn tối đa 1 MiB, timeout 180 giây. Không cài hay nâng cấp package.
5. Chạy subprocess offline để kiểm tra CUDA và import TSR bằng môi trường TripoSR hiện có. Chỉ sau khi qua các kiểm tra mới ghi `triposr.setup.json` với `inference_verified=false`.
6. Gọi launcher Hotfix 1.0.1 bằng **Mode Run, Models triposr**. Run không gọi Setup và đặt HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE. Các guard ổ vật lý/RAM/VRAM vẫn giữ nguyên. Không khởi động Hunyuan hoặc DiffLocks.

D còn khoảng 7.7 GB có thể qua ngưỡng Run 5 GiB của runner nếu lúc chạy dung lượng thực vẫn đủ; không cam kết inference sẽ thành công hoặc không hết VRAM. Không hạ ngưỡng Setup/All 10 GiB và không dùng chúng trong đường chạy này.

## Kết quả cần phân biệt

`TRIPOSR_CACHED_SETUP_READY` chỉ là hoàn tất kiểm tra môi trường và asset.
Chỉ `INFERRED_NATIVE_MESH` trong report/result mới xác nhận đã tạo mesh. Native hair mesh chưa được chứng minh fit đầu, có chất lượng phù hợp hoặc in được. Khi gặp lỗi, cache/env/subject được giữ; đọc report và triposr.log mới.

Bộ test CPU đi kèm kiểm tra liên kết không tạo bản sao, giữ file xung đột/symlink hỏng, kiểm tra SHA256, không ghi setup thành công khi import lỗi, và subprocess smoke offline. Chưa chạy GPU hoặc PowerShell/WSL trên máy người dùng trong môi trường phát triển.
