# Đối chiếu sửa mặt/cổ local — 2026-11-05

Đã có bản mới để xem trước khi tích hợp API. Mục tiêu là giảm cổ dạng ống, mảng tam giác, lệch màu và thử giảm độ hở môi; không xác nhận độ giống ảnh, biểu cảm hoặc khả năng in.

## Xem kết quả

Nếu server review đang chạy, mở lại hoặc Ctrl+F5:

- Trang đang dùng: `http://127.0.0.1:8092/review-site/` — ba lựa chọn mới nằm đầu danh sách.
- Bộ đối chiếu riêng: `http://127.0.0.1:8092/natural-review-20261105/review-site/`.
- Ảnh các góc và ảnh đầu vào: `http://127.0.0.1:8092/natural-review-20261105/review-site/comparison.html`.

Nếu cần khởi động server, chạy `./START-CUSTOMIZATION-REVIEW.ps1` từ thư mục AI. Đây là server local, không cần ngrok. Chưa thay model hoặc cấu hình của AI API/BE/FE sản phẩm.

Có bốn lựa chọn đối chiếu:

| Lựa chọn | Nội dung |
| --- | --- |
| Sửa mặt/cổ — mũ + kính | Cổ mới, cân màu cổ và bản thử môi nhẹ |
| Chỉ sửa cổ — mặt giữ nguyên | Đối chứng giữ hình dạng/texture mặt trước khi sửa môi |
| Trước sửa | GLB gốc từ trang review cũ |
| Sửa mặt/cổ — tóc + kính | Cùng đầu sửa mới trên body/tóc của lựa chọn tóc |

Những bản tăng toàn đầu khoảng 9,65% được giữ trong các attempt cũ, nhưng không đưa vào bộ mới vì có vùng chạm/lộ sát mũ. Bộ mới giữ pose/kích thước đầu như bản review cũ và sửa hình học cổ phía dưới.

## Các nguyên nhân và thay đổi

1. Pipeline cũ bó cổ semantic vào ellipse cố định, sau đó bản đội mũ nâng đầu thêm 18 mm. Khi ghép vào body gốc, điều này làm đường cổ và điểm gắn chưa phù hợp. Bản mới lấy cổ của chính đầu RDF của job, chỉ thu mềm phần chân cổ vào khe cổ áo thật; bỏ connector màu phẳng cũ.
2. Một số vertex ở biên chân cổ/đầu vai không nằm trong mask `neck` của FLAME. Chỉ chỉnh `neck` để sót hàng biên, tạo mép da nằm ngang trên cổ áo. Bản mới bổ sung các vertex `boundary` phía thấp; không chọn nhầm boundary ở mắt/miệng hoặc các vùng cao của mặt.
3. Vùng cổ chứa màu chiếu từ nhiều góc, bóng và phần điền suy đoán. Bản mới dùng parser đã pin để chọn mẫu da cổ quan sát được, fit trường màu bậc hai có trọng số robust, rồi blend riêng ROI cổ. Màu cổ vẫn là ước lượng; chưa phải albedo đã tách ánh sáng. Bản chỉ sửa cổ giữ nguyên pixel texture ngoài ROI cổ.
4. Độ hở môi đã lệch trước khi ghép body. Hai mốc môi trong trên ảnh chính diện cách khoảng 0,395 px; mesh RDF chiếu ra khoảng 7,178 px. Đây là đối chiếu FLAME68/MediaPipe gần đúng, không phải ground truth giải phẫu. Bản thử giảm xuống khoảng 4,421 px, còn chưa khớp ảnh gốc.

Bản thử môi là biến dạng cục bộ theo trục dọc, có regularization Laplacian và vùng bảo vệ mắt/mũi/tai/cổ. Thay đổi 415 vertex quanh môi, lớn nhất khoảng 0,519 mm trong hệ tọa độ body. Góc pháp tuyến thay đổi tối đa khoảng 5,14°, độ dài cạnh thay đổi tối đa 7,53%, không có tam giác đảo pháp tuyến theo guard. Chiếu lại texture vùng quanh môi từ ảnh chính diện; hình học và texture ngoài ROI môi giữ nguyên so với bản nền đã sửa cổ. Không thay beta identity, không tạo biểu cảm cười mới; đây là post-fit cục bộ, chưa phải bộ fit biểu cảm đã kiểm chứng. Chưa kiểm tra tự giao nhau của môi độc lập.

## Dữ liệu, RAM và bảo toàn model

- Bản nguồn mới nhất: `runs/neck-natural-review-20261105/final-low-memory-02/`.
- Head sidecar dùng chung: `head-only.glb` trong thư mục trên.
- Model/ảnh được đưa vào viewer: `runs/customization-prep-20261105/natural-review-20261105/`.
- Bản cũ, attempt trung gian và backup metadata điều hướng vẫn được giữ; không ghi đè GLB nguồn.
- Model đầy đủ được ghép bằng cách append buffer của đầu mới và đổi mesh được tham chiếu bởi node đầu. Dữ liệu nhị phân của thân/mũ/kính được giữ nguyên; bỏ tham chiếu connector cũ. Các buffer của mesh cũ chưa dùng có thể còn trong file debug, nên chưa coi kích thước composite là ngân sách tải cuối cùng.
- Ảnh kiểm tra dùng proxy chỉ bỏ tam giác hoàn toàn dưới Z=0,65 m trong hệ làm việc, nằm ngoài khung cận cảnh. Không decimate hoặc thay vertex/UV trong vùng hiển thị. File GLB viewer vẫn có toàn thân.
- Render Blender vẫn giữ guard tối thiểu 2 GiB trước khi chạy, dừng khi RAM trống xuống dưới 768 MiB. Một số attempt đã bị dừng bởi guard và không được công bố. Bước đóng gói theo luồng/crop bằng CPU có ngưỡng riêng 1,2 GiB vì không dựng toàn bộ scene bằng loader tổng quát.

Mẫu tóc low-poly/hairline, vùng mask tay chạm mũ, vật liệu phụ kiện và pose của body vẫn là vấn đề riêng từ bộ thử trước. Bản này không âm thầm sửa những phần đó. Tóc/mũ trên hai preset có bàn tay khác nhau như trước; cần thống nhất body/preset trước khi tích hợp custom.

## Kiểm tra và giới hạn

- 20 test liên quan: cổ, môi, fit kính và xử lý GLB theo luồng.
- Guard biến dạng môi pass; ngoài ROI được bảo vệ bằng phép so sánh mảng.
- Có kiểm thử GLB thay đầu giữ byte buffer nguồn và giữ tọa độ thân; proxy giữ UV, texture, vertex color và đầu khi round-trip.
- Full suite cuối: xem `runs/neck-natural-review-20261105/full-tests-04.log`; các lỗi môi trường có sẵn gồm thiếu launcher trong manifest, hai trường hợp im2 yêu cầu WSL và thiếu `pyhocon`. Không coi full suite là pass.
- Architecture: xem `architecture-04.log`.
- Đã render trước/sau nhiều góc và kiểm tra mắt thường các bản chính. Chưa có đánh giá độ giống ảnh độc lập, nhiều người/ảnh khác hoặc chứng nhận in.

Code: `tools/customization/neck_natural_math.py`, `neck_natural_trial.py`, `lip_natural_trial.py`, `review_glb.py`, `build_final_low_memory.py`. Các tọa độ gắn cổ và ngưỡng hiện dành cho body/preset đang thử. Không áp dụng mù cho body khác hoặc ảnh có miệng đang mở; bước môi có gate so độ hở trên ảnh và mesh.
