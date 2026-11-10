# Checklist bàn giao từ member AI

Đối chiếu source commit `23b32f5` và các `.gitignore` ngày 03/11/2026. Mục tiêu là tái lập đúng một case ghép mặt vào tượng mẫu trên laptop Windows, không cần bàn giao mọi nhánh nghiên cứu.

## Vì sao pull source chưa đủ

Root `.gitignore` bỏ `/GLB/`, `models/`, `vendor/`, ảnh, runs/outputs, `*.local.json`, `*.glb`, `*.blend`, `*.obj`, `*.npz`, `*.npy` và model weights. Đặc biệt `assets/chibi_templates/.gitignore` chứa `*` và chỉ giữ chính `.gitignore`: **template.json và annotation JSON trong thư mục đó cũng không lên Git**, không chỉ mesh/binary. Vì vậy push lại code không tự làm xuất hiện template.

Giữ quy tắc không đưa ảnh cá nhân, runs, weights, credentials hoặc environments vào Git. Asset cần cho runtime có thể được bàn giao qua thư mục dùng chung hoặc gói asset riêng có manifest; source archive vẫn chỉ chứa code.

## A. Nên push lên Git

1. Đổi component thư mục `new_face ` thành `new_face` (bỏ dấu cách cuối), cập nhật mọi tham chiếu và source manifest liên quan có chủ đích. Đây là điều kiện để checkout bình thường trên Windows.
2. Config mẫu không có secrets/đường dẫn máy riêng: ví dụ `mica.example.json` mô tả schema, relative layout, asset hashes/upstream commit; bảng cách tạo `mica.local.json` trên máy mới. Không commit nguyên config local hoặc giả rằng thay tên file là đủ chạy.
3. Tham số chính xác của case đã chạy: `--source-mode front/three`, source kind, face/pose model và hash, Blender, source units/up/front, template ID và GLB checksum, các quality/warp/color settings có override. Nếu là default code thì ghi commit + tên config, không cần phát minh thêm config.
4. Requirements/lock theo từng môi trường: Python validation/HeadBodySplit, worker MICA, Blender; Torch/CUDA/InsightFace/chumpy cùng các sửa tương thích thực tế. Không chỉ gửi pip freeze của một env không rõ dùng cho bước nào; không gửi nguyên `.venv`.
5. Runbook có lệnh thật tái lập từ checkout sạch đến review, cách nhận/mount assets và lựa chọn interpreter theo stage. CLI hiện hardcode `.venv/bin/python`; native Windows cần cấu hình lại, không chỉ đổi `--blender`.
6. Ghi rõ ứng dụng nào là entrypoint cho demo: `scripts/run_face_on_glb.py` khác với HeadAPI đầu RDF. Nếu member đã viết API gọi ghép tượng, cần cả code/schema/config mẫu của API đó.

## B. Bắt buộc bàn giao riêng cho nhánh template đã duyệt

Lấy **đúng một GLB mẫu** member đã dùng và **toàn bộ thư mục base template tương ứng**, giữ cấu trúc và checksum. Không chỉ gửi ảnh preview hoặc một mình template.json.

Code `scripts/prepare_subject_template.py` hiện đọc trực tiếp:

- `template.json` có status ready và source_glb_sha256 khớp GLB.
- `baseline.npz`.
- `chart_draft.npz`.
- `template.blend`.
- `target_landmarks.json`.
- `source_aligned.npz` của base template, dùng kiểm topology nguồn.

Để chạy trực tiếp một template đã bind cho subject, CLI còn dùng `alignment_draft.json` và toàn bộ đường dẫn trong `pinned_files`, cùng `source_glb`, `source_projection`, `source_texture` và source-head hash. Do đó nên gửi cả folder template và mọi file mà manifest tham chiếu, không cắt bớt theo danh sách tối thiểu bên trên.

Yêu cầu kèm manifest asset: relative path, size, SHA256, template ID/version, nguồn file, file nào phải tải từ tác giả và file nào được bàn giao nội bộ. Template/blend có link hoặc texture ngoài thì phải pack hoặc gửi đủ dependencies. Symlink phải resolve thành file thực trong gói bàn giao; không giữ link chỉ tồn tại trên máy member.

Đường dẫn `/home/...` trong JSON không dùng được nguyên trạng trên máy Windows. Cần bước rebind/relocate rõ ràng và kiểm tra lại hashes, không bỏ pin checks hoặc tính lại hash để hợp thức hóa file khác.

## C. Asset model và inference

- Face Landmarker `.task`: phiên bản/nguồn/hash. Máy này đã có một bản trong D:/AI-cache/mediapipe, cần đối chiếu chứ chưa cần tải lại.
- Pose Landmarker `.task` chính xác member dùng: hiện chưa thấy trong cache đã kiểm tra.
- MICA source commit và checkpoint; FLAME2020 model, landmark embedding, FLAME masks, SCRFD detector và hash. Các file này nhiều phần đã có ở repo Head Reconstruction, nên ưu tiên đối chiếu/tái sử dụng, không yêu cầu push weights lên Git.
- Hướng dẫn dựng MICA worker đúng phiên bản. Config cũ trên máy này trỏ `.venv-mica` không còn tồn tại; đây là dependency cần khôi phục, không phải chỉ thiếu weight.
- SMPL-X chỉ cần nếu demo dựng body từ ảnh; không yêu cầu tải/setup nó cho luồng tượng mẫu. Tương tự không cần toàn bộ STL tóc khi giữ tóc của GLB mẫu.

## D. Một case tham chiếu để kiểm chứng

Nhờ member bàn giao riêng một bộ ảnh được phép dùng, đặt front/left/right đúng tên; GLB đầu ra, preview chính diện/hai bên, report và exact command/commit/config tương ứng. Các output là bằng chứng baseline để so sánh, không đưa vào code Git. Không chỉ gửi một model kết quả mà thiếu đầu vào/cách chạy.

Nếu muốn thử riêng bước ghép trước khi setup inference, cần thêm:

- `head.obj` MICA nguồn đúng topology, units mm, up Y, front +Z theo CLI.
- Folder color của cùng subject có `embedding.npz`, `projection.npz`, `texture.png`.
- Template/GLB base ở mục B.

Đây là nhánh tái lập ghép từ đầu đã có, không được báo là đã chạy ảnh → tượng end-to-end.

## Lỗi Windows cần lưu ý khi bàn giao

- Tên thư mục có dấu cách cuối chặn Git checkout.
- Interpreters Linux và Blender `/snap/...` cần tham số/cấu hình tương ứng Windows.
- `prepare_subject_template.py` dùng symlink_to; trên Windows cần quyền/Developer Mode hoặc phương án copy có kiểm hash. Chưa thử bước này để kết luận lỗi runtime trên máy.
- Test HeadBodySplit gặp WinError 32 khi np.load chưa được đóng trước lúc xóa folder tạm; đây là lỗi test/file-handle đã quan sát, không phải GPU.

## Tin nhắn gợi ý gửi member

> Bạn gửi giúp mình bộ tối thiểu để tái lập case ghép mặt nhé: đúng GLB mẫu + toàn bộ folder base template đã ready (kèm NPZ/Blend/annotation và các file manifest tham chiếu), face/pose model version+hash, config mẫu không có secrets, requirements từng env và lệnh chạy chính xác theo commit. Template hiện bị assets/chibi_templates/.gitignore chặn hết nên chỉ push code sẽ chưa đủ; asset có thể gửi riêng kèm SHA256. Bạn sửa giúp tên folder new_face đang có dấu cách cuối để Windows checkout được. Nếu có thể, gửi thêm head.obj + folder color cùng một case và model/report/preview đã chạy để mình thử riêng bước ghép trước, sau đó mới chạy từ ảnh. Không cần gửi .venv, mật khẩu/token hoặc push weights/ảnh lên Git.
