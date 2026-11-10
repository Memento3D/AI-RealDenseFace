# API ghép đầu cá nhân với bộ phận thử nghiệm

Policy opt-in: `graduation_modular_preview_v1`. Không dùng nhánh Production vá mặt.

## Khởi động để kiểm tra trên FE

Tại thư mục `Memento3D-Head-Reconstruction`:

```powershell
.\scripts/api/START-HEADAPI-ASSEMBLY.ps1 -CheckOnly
.\scripts/api/START-HEADAPI-ASSEMBLY.ps1
```

Launcher dùng `configs/headapi-assembly.json`, cổng mặc định 8090 và thư mục job mới `runs/headapi-assembly`. Không khởi động thêm instance nếu cổng đang được dùng; restart phiên AI của bạn với launcher này. `-CheckOnly` kiểm tra key đã có và prerequisites/pins, không chạy inference hoặc mở server.

Kiểm tra `GET http://127.0.0.1:8090/health`: `backend_ready=true`, `pipeline_policy=graduation_modular_preview_v1`. Key AI vẫn phải khớp `AI_API_KEY` của BE; không đặt key ở FE.

Giữ nguyên hợp đồng BE: `POST /jobs` nhận job/attempt, profile `default`, front/left/right và back tùy chọn; poll và download GLB theo đúng attempt như trước. FE tiếp tục lấy delivery URL từ BE. Dùng scan/job mới trên FE để chạy pipeline mới; mở đơn hoặc output cũ sẽ tiếp tục hiện model cũ.

## Chuỗi đã nối

1. RDF dựng geometry của từng job mới và kiểm tra fitting gates.
2. Hair-cap/mask và strict UV chạy theo luồng hiện có. Appearance mới ước lượng da vùng khuất từ da quan sát của chính job, rồi vẽ chân tóc liên tục theo mốc tai/gáy bằng PCHIP và khoảng cách geodesic; giữ pixel quan sát ngoài vùng thiết kế tóc. Đây là phần được tổng quát hóa từ thử nghiệm buzz-framework, có provenance riêng cho vùng da ước lượng và tóc preset.
3. Căn toàn bộ đầu mới theo eye anchors và scale đồng nhất của preset tốt nghiệp. Không lấy đầu/texture của người trong thí nghiệm làm kết quả mới.
4. Giữ body, tay đầy đủ và mũ/tua từ bản modular hand attempt-03. Bỏ đầu cũ, tóc procedural cũ và strip cổ cũ.
5. Điều chỉnh riêng vùng cổ theo semantic mask, cắt đáy và dựng strip nối vào vòng cổ thân. Các điểm ngoài mask cổ giữ nguyên sau similarity.
6. Fit tóc thư viện từ candidate cage-lift của male-hair-refined-01 theo scalp bounds đầu mới; cắt phần tóc bị che ở mặt trên mũ. Đây là hairstyle preset, không suy luận kiểu tóc cá nhân từ ảnh.
7. Blender giảm lưới body, giữ geometry đầu/tóc/tay/mũ và giảm texture ngoài đầu cho bản web. Head atlas giữ nguyên pixel.
8. Audit decoded GLB kiểm tra đủ component, protected geometry, UV/atlas đầu, texture nhúng, không URI ngoài và dung lượng dưới 50 MiB. Chỉ khi đạt mới publish `render/scene.glb` và trả done.

API dùng stage `04_scene_export` cho phần lắp ghép. Chi tiết nằm ở `render.compose.log`, `render/web-export.log`, `render/assembly-report.json`, `render/report.json` và `render/assembly-config.snapshot.json` của job. Head master dùng để đối chiếu ở `render/assembly-master.glb`; BE nhận bản web, không nhận master.

Các asset tham chiếu được khóa SHA-256 trong config. Nếu file bị sửa/mất, preflight từ chối; không tự nhận asset mới hay tự đổi pin. Dữ liệu và model cũ vẫn giữ nguyên. Quay về nhánh đầu + cap cũ bằng `scripts/api/START-HEADAPI-DEMO.ps1`.

## Phạm vi và kiểm chứng

Đây là bản preview để đánh giá trên FE, `ready_for_machine=false`. Tóc fit theo bounds chưa phải anatomical cage fit cho mọi đầu; tạo hình lọn tóc của asset còn cần đánh giá mỹ thuật. Mối nối cổ là component riêng; màu cổ và da vùng khuất được ước lượng. Chưa kiểm chứng đầy đủ va chạm tóc–mũ/tay–mũ, độ dày, union, watertight hoặc in thực tế. Các chỉnh sửa seam/ear chọn riêng cho từng mẫu trong script nghiên cứu khác chưa được tự động áp dụng cho người mới.

Artifact replay dùng job đã dựng trước để kiểm tra assembly/export, không phải inference mới. Các lần thử và log ở `runs/assembly-api-rehearsal/`. Không coi replay là bằng chứng E2E web → Spring → AI → Cloudinary; lượt đó cần chạy từ scan mới trên FE.

Kiểm tra mã nguồn:

```powershell
.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p test_assembly_pipeline.py -v
.venv\Scripts\python.exe -B scripts/check_architecture.py
```

Full suite còn các lỗi đã ghi nhận ở workspace resume, hair launcher manifest, im2 Windows/pyhocon; không tuyên bố toàn suite đạt.
