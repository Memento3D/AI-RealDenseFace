# Im2Haircut field transform update 0.3

## Mục tiêu của lượt thử này

Giảm việc chạy GPU lặp lại: dùng lại baseline 50/300 đã có, kiểm tra chân sợi và preprocessing trên CPU, tạo một dataset riêng rồi chạy tối đa một worker 50 epoch. Không chạy lại hai phép thử Gabor/depth của update 0.2. Không tải model hoặc cài thêm môi trường.

Căn cứ hiện có:

- Các ảnh review cho thấy nhiều cấu hình cùng tràn vùng trán; chưa chứng minh nguyên nhân là camera, scalp hay hướng sợi.
- Chân sợi bản 50 và 300 đã gửi gần như trùng nhau (sai khác tối đa khoảng 5e-9 đơn vị native khi đối chiếu OBJ/NPZ). Script mới tính lại điều này từ NPZ tại máy bạn. Đây không phải chứng nhận vị trí giải phẫu đúng.
- Source alignment upstream warp map góc như ảnh xám và warp GB như màu ảnh. Nội suy 1° và 179° thành 90° không đúng với hướng vô hướng có chu kỳ 180°. Việc biến đổi vector theo affine cũng chưa được xử lý ở đường đi này.
- Góc xoay của case 04 nhỏ. Sửa phép biến đổi không được coi là đã giải quyết toàn bộ tóc sai, tóc mai hoặc vùng phía sau.

## Cài bản cập nhật

Giải nén source vào project hiện tại, thay thế các file code. Giữ nguyên `input`, `runs`, `vendor`, weights và các môi trường. Không chạy từ một thư mục project mới thiếu kết quả đã có.

Profile `configs/im2.review04.json` chứa đường dẫn các run đã biết và ma trận affine của case 04. Script bắt buộc ma trận này tái tạo đúng từng pixel ảnh RGB và Gabor legacy đã căn. Nếu ảnh/dữ liệu thay đổi hoặc khác run, script dừng trước GPU, không suy đoán đường dẫn mới và không tải lại landmark model.

## Một lệnh trong Ubuntu

```bash
cd /mnt/d/FPTU/07/EXE201/ai-3d-generation/Memento3D-Head-Reconstruction
bash scripts/test_im2_fix04.sh
```

Lệnh tự kích hoạt môi trường `im2haircut`. Không cần `pip install -e .` hay chạy lại các bước setup.

Nếu chỉ muốn kiểm tra CPU, không chạy GPU:

```bash
bash scripts/test_im2_fix04.sh --cpu-only
```

## Lệnh thực hiện những gì

1. Xác minh ảnh gốc, map đã chuẩn bị, camera, pin upstream, SHA256 source/weights và liên kết các baseline cũ với dữ liệu.
2. Dùng lại baseline 50/300 để xuất ba ảnh: chân sợi nhìn thấy, đoạn đầu sợi (5% số điểm), toàn bộ sợi chiếu lên ảnh. Ghi mức dịch chuyển chân sợi giữa các bản.
3. Tái tạo alignment RGB và map góc cũ bằng ma trận đã lưu. Nếu không khớp tuyệt đối thì dừng và ghi lỗi rõ ràng.
4. Tạo dataset mới `~/memento-hair/Im2Haircut/data/fieldfix04_<UTC time>_<id>/`. Giữ byte ảnh, mask, camera, body mask, confidence và depth. Chỉ thay hai map hướng.
5. Gabor: biến đổi hướng trong hệ ảnh, nội suy cos(2θ)/sin(2θ) có mask, mã hóa lại đúng chu kỳ. HairStep: biến đổi vector hướng GB, nội suy có mask và chuẩn hóa. Giữ nguyên kênh R. Nếu trung bình vector bị triệt tiêu thì dùng hướng nguồn gần nhất đã biến đổi; không tự đặt hướng về 0.
6. Kiểm tra dataset thật của upstream đọc được trên CPU và mọi tensor hữu hạn.
7. Chạy đúng một worker 50 epoch từ cùng pretrained weights, cùng camera/mask và loss weights baseline. Giới hạn upstream này tương ứng 51 updates. Không resume optimizer, không retry tự động và không bật upsampling.
8. Xuất cùng bộ ảnh chẩn đoán cho bản mới, report và HTML để so trực tiếp với baseline. Worker giữ nguyên cơ chế log, RSS/RAM, VRAM toàn GPU, nhiệt độ và công suất.

## Kết quả cần mở

Đường dẫn được in cuối terminal:

```text
runs/im2_fix_<UTC time>_<id>/
  review.html
  experiment.report.json
  fields_before_after.png
  baseline_50/
    roots.png
    early_segments.png
    overlay_r1.png
  baseline_300/
    ...
  field_fix_50/                  # chỉ có khi worker thành công
    ...
  worker_field_fix/
    memento_im2_.../
      pipeline.report.json
      inference.log
      metrics.csv
      export/hair.strands.npz
```

Gửi `experiment.report.json` và ảnh trang review khi xong. Có thể kiểm tra lại kết quả hiện có mà không cần chạy bộ ablation cũ.

## Giới hạn cần giữ rõ

- HairStep GB dùng giả thuyết hệ ảnh x sang phải / y xuống dưới, nhất quán với đích chuyển đổi Gabor trong upstream nhưng chưa được hiệu chuẩn độc lập trên tóc thật. Nó là một giả thuyết thử nghiệm được ghi trong report.
- Cả hai map hướng cùng thay đổi trong một lượt. Nếu tốt hơn thì chỉ kết luận recipe kết hợp có ích trên case này, không xác định được đóng góp riêng từng sửa đổi.
- Sửa circular interpolation không khôi phục chi tiết Gabor bị thiếu do ảnh ít texture hoặc độ phân giải thấp. Không dùng map đã làm mượt để giả vờ có thêm thông tin.
- Không dịch/chỉnh/prune chân sợi, không fit lại camera, không đổi hình học scalp, không tạo tóc mai hoặc gáy bằng quy tắc thủ công. Màu đỏ của chân sợi ngoài mask không tự chứng minh scalp sai: chân sợi có thể nằm dưới hoặc sau bề mặt tóc.
- Ảnh trái/phải/sau chưa tham gia loss. Không ghép sang FLAME, không phục hồi màu tóc, không xếp hạng chất lượng tự động hoặc tuyên bố tóc đã hoàn chỉnh.
- Pixel support vẫn là proxy sợi thưa, không phải render Gaussian/Blender cuối. Độ dày pixel, che khuất và mật độ đều ảnh hưởng chỉ số.
- RSS là tổng theo nhóm tiến trình; VRAM là toàn GPU. Không dùng các số này như lượng VRAM cấp phát chính xác cho riêng pipeline.

## Kiểm chứng của bản phát hành

CPU: full unittest discovery, kiểm tra kiến trúc, parse cú pháp Python 3.9 và `bash -n`. Tests mới kiểm tra wrap 1°/179°, xoay vector 90°, bảo toàn kênh R và dữ liệu ngoài mask, từ chối affine/raw map không khớp, dataset gốc không đổi và giới hạn zero/one worker khi CPU-only hoặc worker lỗi. Test orchestration dùng worker mô phỏng; không phải kết quả GPU thật.

Bản sửa chưa được chạy trên camera/map riêng của bạn trong môi trường build. Không có kết quả GPU hay hình ảnh cải thiện được dựng sẵn trong ZIP.
