# Tạm dừng thử nghiệm đầu trọc và kế hoạch tiếp tục

Ngày ghi nhận: 03/11/2026. Theo yêu cầu của người dùng, tạm dừng cải tiến đầu trọc để đánh giá pipeline AI Production của team trước. Không xóa dữ liệu, không thay model pins, không đổi API, không pull hoặc cài dependency ở lượt ghi chú này.

## Baseline cần giữ

- Trang tổng hợp: `runs/experiments/head-review-package-01/review.html`.
- Bản được chọn: `runs/experiments/ear-texture-03/{input04,web-capture,capture02}/fill_only/head.glb`.
- Bản sao đã kiểm chứng và preview nhiều góc: `runs/experiments/head-review-package-01/{case}/`.
- Manifest và kiểm tra: `head-review-package-01/manifest.json`, `ear-texture-03/audit.json`; mọi đường dẫn trên nằm dưới `runs/experiments/`.
- Chuỗi lựa chọn: bald-combined-01 (tone mild) → bald-seam-band-04 (gentle) → neck-semantics-01 (neck_balance) → neck-completion-01 (clean) → ear-texture-03 (fill_only).
- Đây là xử lý appearance trên RDF/UV đã có, không phải chạy mới toàn bộ ảnh → reconstruction → API. API/web vẫn dùng cấu hình demo cũ; đừng nhầm baseline này với model đang được backend phục vụ.

## Đã làm và giới hạn bằng chứng

Đã phủ màu ước lượng cho da đầu, làm dịu đường nối texture, xử lý cổ theo semantic class và bổ sung màu tai từ observations cùng tai với template-mask/distance/normal guards. Bản ear fill_only giữ nguyên pixel known của bước cổ; riêng bước cổ clean đã thay một phần màu cổ quan sát và đánh dấu lại là estimated. Không được diễn giải rằng toàn bộ chuỗi giữ nguyên mọi màu quan sát.

Giữ geometry RDF gốc, GLB có texture nhúng; checksums và provenance/confidence được giữ. Màu vùng khuất là ước lượng, không phải chi tiết người thật được phục hồi. Chưa khử ánh sáng để thu albedo; chưa đo độ giống bằng ground truth; kính và tóc chưa giải quyết. Bản hình học còn miệng cổ mở, chưa xác minh self-intersection, độ dày tối thiểu, supports/slicing hoặc in resin thực tế. GLB đẹp không đồng nghĩa mesh in được.

Quan sát hiện tại: da đầu/cổ còn phẳng hơn mặt; còn đường chuyển màu ở thái dương, trước tai và cạnh cổ. Cần phân biệt lỗi texture/alignment với lỗi geometry bằng bản xám và các góc trên/dưới cằm.

Kết quả suite gần nhất trước khi ghi chú: 535 tests, 2 failures, 4 errors, 1 skipped; architecture PASS. Log: `runs/experiments/head-review-package-01/unittest.log`. Các nhóm lỗi: root feature manifest; im2 field-fix hai subcases; im2 review; hair-studio cache invalidation; workspace/figurine mocked resume. Không ghi suite PASS.

## Khi quay lại nhánh thử nghiệm này

1. Kiểm tra RAM trống, source hashes và baseline manifest trước khi chạy. Worker tuần tự: bắt đầu tối thiểu 2 GiB RAM trống; ngừng child dưới 768 MiB hoặc timeout; không đóng ứng dụng người dùng và không giảm guard để ép chạy.
2. Khoanh mask chuyển tiếp ở thái dương/cổ bên trong không gian bề mặt 3D. Bảo vệ mặt trung tâm, mắt, môi; không làm mờ toàn mặt hoặc tăng confidence vùng inferred. Chạy một thay đổi mỗi attempt mới, giữ baseline nguyên trạng.
3. Đối chiếu ba bộ ảnh, góc chính diện/profile/đỉnh/dưới cằm, bản màu và bản xám. Metric đường nối chỉ là chẩn đoán, không phải bằng chứng độ giống; tránh chọn chỉ theo metric của chính vùng đang fit.
4. Nếu appearance đạt trần do ảnh thiếu/ánh sáng/geometry, quay lại nghiên cứu reconstruction hoặc acquisition; không tiếp tục che lỗi bằng chỉnh màu. Fine-tune chỉ sau khi xác định failure mode, dữ liệu, quyền sử dụng và tiêu chí đánh giá.
5. Sau khi chọn baseline mới mới thiết kế orchestrator versioned cho toàn chuỗi, rồi chạy API với job/attempt mới. Orchestrator `bald_combined_trial.py` hiện chỉ bao ba bước đầu và labels tới 8; chưa bao seam/neck/ear mới. Không sửa allowlist để chấp nhận tùy tiện labels 9–15.
6. Kiểm chứng in là nhánh riêng: scale/unit, đóng cổ hoặc ghép body, manifold/watertight, giao cắt, features mỏng, orientation/supports và slicer; cuối cùng cần in thử. Không thể xác nhận in thành công chỉ bằng kiểm tra mesh.
7. Sau thay đổi code: chạy unittest discover -s tests -v và scripts/check_architecture.py; ghi cả lỗi còn tồn tại. Không đưa runs, ảnh, weights, credentials hoặc environments vào Git.

## Đối chiếu AI Production trước khi chuyển hướng

Repo local: `D:/FPTU/07/EXE201/ai-3d-generation/Memnoto3D-AI-Production`.

Snapshot chỉ đọc ngày ghi nhận:

- Branch: `feature/DonVT-local-gpu-pipeline`, HEAD `0623718` (`fast api`).
- Có untracked `downloaded-model.glb` và `runs/api/`; giữ nguyên, không clean/reset/stash bừa.
- Tracking refs là dữ liệu local có thể cũ; chưa fetch, chưa xác minh code mới remote hoặc branch của member.
- README local mô tả phase 1 single-image Hunyuan geometry/texture, không phải bằng chứng luồng template body + face đã triển khai.
- API local nhận một `input_url`; Head Reconstruction demo dùng các ảnh front/left/right. Không thay service chỉ vì cùng có endpoint jobs.
- Profile `configs/phase1.rtx4050.full.json` khai báo min_vram_gib 5, min_ram_gib 12, cpu_offload true, texture/render 512, steps 20. Đây là cấu hình cần kiểm thử, không phải số đo hiệu năng hoặc bằng chứng chạy được trên laptop.

Thứ tự đề xuất khi bắt đầu kiểm thử repo team:

1. Xác định branch/commit mà member muốn demo. Fetch refs trước; xem diff/log/config/schema so với HEAD local. Không mặc định pull branch hiện tại là lấy đúng code của member.
2. Nếu đúng tracking branch, không divergence và không va chạm local artifacts thì fast-forward có kiểm tra; nếu branch khác hoặc cần giữ môi trường demo, dùng checkout/worktree riêng và output mới. Không merge vào main hoặc chỉnh thuật toán member trước khi có baseline.
3. Đọc AGENTS và hướng dẫn mới sau fetch; kiểm tra model pins, weights/cache, Python/CUDA/WSL/native requirements, disk/RAM/VRAM trống trước khi tải hoặc cài. Tách venv với repo Head Reconstruction để tránh phá môi trường đang demo được.
4. Chạy tests/health/preprocess hoặc smoke nhỏ trước. Nếu có template-body workflow thật: kiểm tra asset body, vị trí cổ, tỉ lệ/trục/units, face/head mesh contract, UV/material và cách ghép.
5. Chạy một case tương đương input04, đo thời gian và tài nguyên thực tế, lưu report/log/commit/config. Nếu model không vừa máy thì dừng và phân tích stage gây thiếu bộ nhớ; không mặc định tải thêm model hoặc thuê cloud ngay.
6. So sánh cùng input: độ giống quan sát, đường nối đầu-cổ/body, texture, mesh integrity, thời gian và khả năng tái chạy. Chọn pipeline theo kết quả demo thực tế, không theo tên repo hay người viết.
7. Giữ pipeline hiện tại làm phương án dự phòng; chỉ nối web khi API contract và một lượt chạy local đã kiểm chứng. Face từ repo này có thể dùng làm module ghép body nếu topology/scale/neck contract phù hợp; không nhất thiết bỏ toàn bộ công việc hiện có.

## Digital Salon và mức ưu tiên

Nguồn chính: https://github.com/digital-salon/Digital-Salon (README được xem lại ngày ghi nhận).

Repo giới thiệu công cụ tạo/chải/mô phỏng tóc AI và vật lý, với thao tác tương tác; README còn yêu cầu head mesh riêng. Đây không phải bằng chứng có sẵn luồng tự động ảnh → khuôn mặt cá nhân → mesh tượng in. Chuyển strands/groom thành khối tóc có độ dày, ghép đầu, kiểm tra mesh và in là các hạng mục bổ sung cần xác minh riêng.

Vì đang ưu tiên đầu trọc và demo trên laptop, hoãn cài/build Digital Salon. Chưa đủ thông tin deadline, môi trường và kết quả thử để cam kết kịp hoàn thiện tích hợp. Có thể đặt một đợt feasibility riêng sau khi pipeline team chạy được: build tối thiểu, mở sample, kiểm tra export thực có, rồi mới quyết định nghiên cứu tóc tiếp.

## Quyết định hiện tại

Đóng băng baseline đầu trọc để tiếp tục được về sau. Ưu tiên lấy đúng phiên bản AI Production của team, kiểm tra local một case và API trước; không tiếp tục mở thêm nhánh tóc trong giai đoạn chuẩn bị demo. Chưa thực hiện fetch/pull/cài đặt/chạy AI Production trong lượt này.

Cập nhật ở lượt tiếp theo: đã fetch main mới, tạo checkout WSL đúng commit và kiểm tra source/tests; Windows checkout bị chặn bởi thư mục có dấu cách cuối tên. Xem [đánh giá local AI Production](AI-PRODUCTION-LOCAL-ASSESSMENT-VI.md) để biết vị trí checkout, kết quả test và các asset còn thiếu. Chưa chạy inference đầu cuối.
