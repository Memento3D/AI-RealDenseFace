# Experiment release — comparison harness

Update 2026-10-27: separate prepared-sample Im2Haircut runner is now included;
see IM2HAIRCUT_UPDATE.md. Raw-photo preprocessing and integration in this
baseline/refined comparison harness remain unimplemented. Historical scope
below describes the earlier comparison release, not the new sample runner.

## Quyết định
Giữ RealDenseFace + FLAME2023 làm baseline mặt. So sánh baseline V1 với texture/eye split V2 trước. Không dùng tóc procedural hoặc tóc mẫu trong phép so này.

Nhánh tóc: shortlist MonoHair (video/multi-view) và Im2Haircut (ảnh, prior + tối ưu). Cả hai CHƯA tích hợp trong bản này, chưa benchmark 6GB. Cần audit code/weights, pin commit, xác nhận preprocessing/camera/FLAME, license và môi trường riêng trước khi xây adapter. Không có lệnh giả chạy nhiều model tóc.

Nguồn: https://github.com/KeyuWu-CS/MonoHair và https://github.com/Vanessik/Im2Haircut . Không cài đè môi trường RDF. Không tải weights tự động ở bản comparison.

## Chạy ngay trên máy đã setup

```powershell
.\.venv\Scripts\python.exe .\scripts\compare_variants.py --input-dir input/04
```

Chạy tuần tự baseline và refined để tránh chồng VRAM. Mỗi variant có run mới, pipeline.report.json và metrics.csv do runner hiện có tạo. comparison.json ghi hash ảnh, lệnh, exit code, thời gian, log và các run mới. Không chạy job khác đồng thời; nếu có nhiều run mới thì association được ghi ambiguous, không tự chọn kết quả.

Hai variant đều dựng lại geometry: đây là so sánh pipeline đầu-cuối, KHÔNG phải ablation texture với geometry cố định. Cần kiểm tra geometry/pose tương đương trước khi quy kết thay đổi do texture.

## Đánh giá
Cùng góc render, ánh sáng, camera, kích thước hiển thị. Đánh giá riêng likeness, mắt/mí, đường nối da, tóc mai/chân tóc/gáy, clipping và khả năng mở GLB. Dùng thêm ảnh giữ lại không dùng fitting cho nhánh tóc. Không chọn model chỉ vì log thành công hoặc ít VRAM.

Đo RAM process tree, VRAM toàn GPU và allocated/reserved nếu backend cung cấp; tách lần đầu và lần chạy lại. Không gọi số đo VRAM toàn GPU là VRAM riêng của model.

## Chưa hoàn thành
Chưa có mắt PBR/giác mạc mới; V2 chỉ là bản refinement đã có. Chưa có benchmark MonoHair/Im2Haircut, chưa có mô hình thắng, chưa có bảo đảm chạy 6GB. Bản này tạo baseline review; không phải bản hoàn tất yêu cầu nhiều model tóc.
