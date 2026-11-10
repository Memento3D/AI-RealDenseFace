# Kiểm tra cập nhật AI Production — 28ea3e1

Ngày: 03/11/2026. Commit được fetch: `28ea3e16fbd94537f51b52c382436e308fe02541`, sau commit bổ sung dữ liệu `7f74848`.

## Đã bổ sung

- GLB mẫu men, men2, woman và một kết quả mẫu; file lớn dùng Git LFS.
- Hai folder template với annotation, mapping, baseline, Blend và manifest.
- Pose Landmarker `.task` có dữ liệu thật: 9.398.198 bytes, SHA256 `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`.
- Config MICA local, đầu MICA/màu nguồn và các kết quả tham chiếu.

Đã đối chiếu cây Git và hash các file manifest tham chiếu. Báo cáo máy đọc được: `runs/experiments/ai-production-review-02/audit.json`. Không tạo archive có ảnh, weights hoặc kết quả cá nhân. Git đã nhận dữ liệu được member commit; chưa checkout toàn bộ dữ liệu vào worktree Windows.

## Các vấn đề còn lại

1. **Windows checkout vẫn bị chặn.** Component `new_face ` chưa bỏ dấu cách cuối; hiện 693 đường dẫn trong tree nằm dưới tên không tương thích. Không thử lại merge biết trước sẽ lỗi, không tắt bảo vệ NTFS. Repo Windows vẫn ở nhánh cũ, các output local giữ nguyên.
2. **Template gốc dùng tên GLB cũ.** `graduation_1118_v1/template.json` tham chiếu `GLB/1118d65dcaf4a47fa12a35b1da819173.glb`, nhưng tree chỉ có men/men2/men2_khoa/woman. Pointer LFS của men.glb khai báo đúng hash mà template yêu cầu: `96496c...db84d`. Điều này cho phép đối chiếu alias sau khi tải và hash dữ liệu thật, không phải lý do bỏ pin checks.
3. **Bốn symlink template Phước trỏ về máy member.** baseline.npz, chart_draft.npz, template.blend, target_landmarks.json dùng `/home/cao-khoa/...`; dữ liệu đích tương ứng có trong template gốc. Cần materialize file thật hoặc relocate links có kiểm hash.
4. **Config MICA còn là cấu hình máy cũ.** Các đường dẫn repo, Python và assets nằm dưới `/home/...`. Worker/env trên máy này chưa được dựng lại. Có thể tái sử dụng checkpoint/FLAME đã pin trong repo Head Reconstruction nếu hợp đồng và môi trường phù hợp.
5. **CLI chưa được sửa portability.** Python `.venv/bin/python`, root `new_face `, symlink_to và mặc định Blender Linux còn nguyên. Blender Windows 5.2.3 đã có nhưng chưa đủ để chạy nguyên luồng.
6. **Đã tải và kiểm payload LFS.** Ba asset tổng 289.843.875 bytes: men.glb 76.531.304 bytes; baseline.npz 128.621.966 bytes; template.blend 84.690.605 bytes. Cả ba khớp size và SHA256 trong pointer/manifest; xem cached_payload_verified trong audit.json. Lượt tải đầu bị ngừng khi chờ lâu; lượt tải lại có timeout mạng ngắn và không retry đã hoàn tất thành công.

## Phạm vi thay đổi code và kiểm tra

Đối chiếu với `23b32f5`: không có thay đổi file Python/TOML cốt lõi sau khi loại vendor/runs/outputs/diagnostics; chủ yếu thêm asset, local config, cache và metadata. Không chạy lại toàn bộ unit suite chỉ vì bổ sung dữ liệu; kết quả suite trước vẫn có phạm vi source review copy của commit cũ. Lượt này kiểm manifest, references, symlink và LFS riêng; chưa inference, chưa tái lập ghép trong Blender, chưa demo web.

Đã chạy native Blender 5.2.3 Windows ở background/factory startup, tắt autoexec, import riêng men.glb: PASS. Chi tiết mesh/UV/material/image ở `runs/experiments/ai-production-review-02/blender-import.json`; log và RAM guard ở blender-import.log/blender-stage.json. Bản GLB runtime trong thư mục ignored chỉ phục vụ kiểm tra; không tạo archive hay commit asset. Đây là bằng chứng đọc model mẫu bằng Blender Windows, không phải bằng chứng transfer/roundtrip/render hoặc printability. Guard bắt đầu >=3 GiB RAM trống, ngừng dưới 768 MiB hoặc quá 120 giây; không đo peak.

Root gitignore hiện chỉ giữ quy tắc venv, dẫn tới cả cache pyc, egg-info, ảnh/output/vendor vào tree. Đề xuất team khôi phục ignore phù hợp và bàn giao runtime assets theo manifest/LFS có chọn lọc. Không tự xóa file hay rewrite lịch sử Git của team.

## Thứ tự tiếp tục

Xác minh ba payload LFS → đổi tên thư mục để checkout Windows được → materialize/relocate template và GLB alias bằng checksum → cấu hình interpreter/Blender → thử ghép từ đầu/màu MICA có sẵn → sau khi đạt mới dựng lại MICA worker và chạy ảnh mới. Chưa cần mở thêm nhánh Body/SMPL-X hoặc tóc cho case tượng mẫu.
