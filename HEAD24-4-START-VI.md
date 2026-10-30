# Head24.4 — tiếp tục bộ05 từ fit đã lưu

## Cách dùng

Giải nén gói Update trực tiếp vào project, ghi đè các file Head24 tương ứng.
Đây là bản bổ sung đầy đủ của nhánh Head24; cần giữ source V2.3, assets tóc,
vendor và hai môi trường Python đang có. Không xóa runs, input hoặc cache OCR.

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN-HEAD24.ps1 -ResumeRun ".\runs\head24_20261029_044424_683718"
```

Lệnh xác minh và dùng lại `01_crops` + `02_fit` từ run gốc, tạo run mới,
chạy lại workspace và figurine trên CPU. Không chạy lại detector hoặc RDF/GPU.
Không dùng run resume gần nhất làm nguồn: nó không chứa đủ hai stage gốc.
Gói Results05 có kết quả đã chạy sẵn: giải nén, mở review.html; model ở selected/.

## Lỗi gốc và thay đổi

1. **Hệ tọa độ sọ.** Gate cũ so sánh trực tiếp hình học front với từng view.
   Chuyển động cổ/hàm và thành phần mắt bị trộn vào kiểm tra đầu trên, gây
   kết luận EXPRESSION_MISMATCH quá rộng. Nay dùng rigid Kabsch trên proxy sọ
   phía sau, chỉ lấy unique vertices và giữ riêng 1/3 điểm để kiểm tra.
   Camera phục vụ tóc được hợp thành với transform rigid; không scale, shear
   hay thay đổi mesh mặt. Camera contract của từng view vẫn được kiểm tra riêng.
   Ngưỡng 5px nay áp dụng cho proxy sọ sau đã căn hệ tọa độ; đây là thay đổi
   phạm vi gate, không phải xác nhận toàn bộ mặt có sai số dưới 5px.
   Phần mặt trước còn biến dạng biểu cảm được báo cáo riêng, không che giấu.
   Proxy là quy tắc hình học theo hệ +Y up/+Z face, chưa phải segmentation giải phẫu.
2. **Lip/eye parser trong ảnh đeo kính, ngược sáng.** Lip mỏng chỉ chạm 4 đỉnh
   nguồn, còn eye labels không có. Workspace mới gắn landmark front thực với
   SHA256 pixel ảnh. Chỉ khi semantic mouth anchor thất bại mới dùng ROI giới hạn
   bởi landmark miệng để chọn đỉnh nguồn đang nhìn thấy. Đây là ROI, không phải
   đường viền môi tái dựng chính xác. Cho phép mắt bị che khi có kính được parser
   phát hiện và observation khớp ảnh; không dựng iris/gaze bị khuất.
   Painting guide giữ ảnh kính, dùng fade mềm và bù màu cục bộ có giới hạn ở ROI
   miệng. Không coi đó là khôi phục albedo hoặc màu da thật.
3. **Tóc mai.** Tip handles phải thuộc vùng biến dạng trước tai đã bảo vệ.
   Sửa lỗi handles nằm ngoài region; không bỏ kiểm tra tai, contact hay volume.
4. **Checkpoint tóc.** Lưu shell đã fit trước khi tinh chỉnh tóc mai, ràng buộc
   hash geometry, mắt, asset, policy và code tạo shell. NPZ có checksum, không
   dùng pickle. CLI run_pipeline_v24.py có --geometry-cache cho thử nghiệm sau;
   không tự dùng checkpoint của người khác.

## Kiểm chứng trên bundle05 thực

Đã chạy toàn bộ các bước CPU sau GPU fit từ bundle người dùng cung cấp, với
parser resnet18.onnx đúng SHA256 cũ. Workspace upload thiếu front observation;
bản chạy kiểm chứng dùng landmark từ diagnostics đã upload trước đó, sau khi
đối chiếu chính xác SHA256 pixel crop front với report fit. Không thay geometry,
q hoặc camera contract. Khi chạy resume trên máy, workspace mới tự nhúng dữ liệu này.

- DEMO_EXPORTED, chọn sideburn_21778_0p4; thời gian CPU host khoảng380 giây.
  Đây không phải benchmark Windows/GPU.
- Gate cũ: raw upper-all P95 left16.92px, right33.75px. Sau căn hệ sọ, holdout
  proxy sọ sau P95 left0.531px, right0.311px; train91 và holdout46 unique points.
  Đây là kiểm tra correspondence nội bộ, không phải độ chính xác so ảnh độc lập.
- STL được chọn: 52.426 đỉnh,104.852 tam giác, một thành phần, kín, winding nhất quán,
  volume dương; kích thước14.494 ×16.524 ×22.895mm. Boundary/non-manifold edges0,
  zero-area faces0. Chưa kiểm toàn bộ self-intersection hoặc độ dày.
- Surface guard200 mẫu vùng mặt trước: max khoảng2.37e-7mm; không phải Hausdorff
  toàn mesh, không gồm hốc mắt/thái dương. Geometry nguồn không bị tạo mặt mới.
- 53 test Head24 qua; architecture check PASS; 63 file runtime V2.3 giữ nguyên SHA.
- Full suite:349 test,1 failure +2 errors giống baseline296 test. Hai lỗi liên quan
  viewer/hair-on-head/index.html thiếu trong source ZIP, lỗi còn lại thiếu pyhocon
  ở runtime kiểm thử. Không tuyên bố toàn bộ suite xanh. Log nằm ở verification/.

## Giới hạn của kết quả này

Mục tiêu đã kiểm chứng là pipeline đi hết và giữ hình học mặt nguồn. Chưa xác
nhận likeness, thẩm mỹ tóc hay màu da. RGB nền107,100,113 còn ám màu ảnh ngược sáng;
kính chỉ có texture, chưa có gọng 3D. Tóc21778 vẫn là template, chưa giống tóc thật
của bộ05. Head/cổ nguồn22mm, chưa có toàn thân/đế80–100mm. STL kín không đảm bảo
in thành công; ready_for_machine vẫn false, chưa slice/hiệu chuẩn resin/in thật.
Support thử nghiệm trong selected/ chưa được kiểm chứng cho máy in.
