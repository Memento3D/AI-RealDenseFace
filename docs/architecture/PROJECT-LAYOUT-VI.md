# Cấu trúc project và phạm vi tái tổ chức

**Cập nhật sau đợt gom JSON:** các manifest code đã chuyển từ gốc sang `configs/manifests`; evidence được giữ trong `docs/legacy` và `maintenance/verification`. `_archive`, `verification`, `validation` cùng bốn thư mục tên lỗi đã được dọn sau khi giữ phần cần thiết. Xem [chi tiết cập nhật](JSON-AND-FOLDER-CLEANUP-VI.md). Các đoạn dưới mô tả đợt chuyển launcher trước đó; README ở gốc là hướng dẫn hiện hành.

## Đánh giá kiến trúc

Phần `src/` đã có hướng tách domain/application/pipelines/adapters và bootstrap. Vấn đề chính của layout là nhiều launcher thử nghiệm cùng nằm ở gốc với launcher demo; tài liệu vận hành cũng nằm lẫn với file build/config. Không có cơ sở coi mọi script cũ là dư thừa: nhiều script được import, gọi qua CLI hoặc nằm trong checksum/cache producer policy.

Đợt này tổ chức lại entrypoint và tài liệu, không viết lại thuật toán, HTTP route, job contract, pipeline policy hoặc model pin. Không chuyển SDK vào domain, không sửa kết quả inference.

```text
START-HEADAPI-ASSEMBLY.ps1  # compatibility wrapper
src/                      # packages, application and adapters
tools/                    # stable module/CLI paths for API and research tools
scripts/
  api/                    # API/demo/assembly launch and API testing
  setup/                  # explicit environment installation
  experiments/            # versioned research launchers
  review/                 # local viewers/probe review
  maintenance/            # operational verification
  *.py                    # existing public imports/cache producer paths
configs/                  # unchanged runtime policies and asset pins
requirements/             # environment-specific dependencies
tests/                    # regression tests
schemas/                  # contracts/schemas
docs/
  guides/                 # moved root-level user/runbook documentation
  architecture/           # structure and migration notes
assets/ input/ vendor/    # local resources, retain existing ignore rules
runs/                     # generated outputs; still contains legacy runtime dependencies
maintenance/              # cleanup reports and local inventories
```

## Các thay đổi

18 PowerShell launcher chuyển từ gốc vào scripts theo vai trò; 6 tài liệu chuyển vào docs/guides. Tham chiếu cũ trong code, hướng dẫn và launcher gọi nhau đã cập nhật. Chỉ giữ wrapper assembly ở gốc vì đây là lệnh demo đang dùng. Những lệnh cũ khác phải dùng đường dẫn mới trong scripts/README.md.

Các launcher mới tính repository root bằng hai cấp cha của PSScriptRoot. Default parameter cần đường dẫn root cũng được cập nhật. Không đổi tên tham số hoặc giá trị mặc định của pipeline.

Hai helper dọn dẹp chuyển từ maintenance sang scripts/maintenance; maintenance giữ report/data. [Danh sách chuyển file và checksum](relocations-20261110.json) được lưu trong Git để người khác có thể đối chiếu mà không cần output cục bộ.

Sáu manifest code ở gốc được giữ đúng vị trí do các phiên bản runtime đang đọc trực tiếp. Các entry launcher đã di chuyển được cập nhật path và SHA-256 tương ứng; đây là đổi checksum code có chủ đích do đổi vị trí/root resolution, **không phải đổi model pin**. Entry khác giữ nguyên, kể cả những thiếu sót đã tồn tại từ trước. Chi tiết old/new và hash cũ ở inventory cục bộ `maintenance/local-cleanup/20261110/restructure-result.json`.

Các file dưới src và configs không thay đổi. Hai thông báo hướng dẫn trong hair_studio.py được cập nhật tên lệnh mới; thuật toán của chúng không đổi. Test đường dẫn launcher được cập nhật để kiểm tra vị trí chuẩn mới.

Một file tên hỏng do shell ở gốc được chuyển vào quarantine cục bộ có ghi lại tên gốc, không coi là script và không thực thi nội dung.

## Kiểm tra và giới hạn

PowerShell AST parse toàn bộ scripts: PASS. Assembly preflight qua wrapper ở gốc và qua launcher canonical từ working directory bên ngoài repo: PASS. Base-only preflight: PASS. Architecture check: PASS. Unittest sau tổ chức lại được lưu trong `maintenance/local-cleanup/20261110/unittest-restructure.log`; kết quả thực tế phải được báo riêng, không coi preflight là kiểm chứng toàn pipeline 3D.

Không chạy cài môi trường, inference tốn GPU, tạo job mới hay push trong đợt này. Các script Python đang có ràng buộc import và code manifest được giữ vị trí để tránh một lần di chuyển hàng loạt làm mất tương thích. Đưa runtime production ra khỏi runs cần migration asset/config riêng, không làm bằng cách xóa hoặc đổi đường dẫn tùy ý.

Kết quả unittest cuối sau di chuyển launcher và helper: 602 tests, 1 failure, 4 errors, 1 skipped; cùng nhóm lỗi so với lượt kiểm tra trước refactor (resume mock, manifest thiếu file và dependency im2mesh). Log cuối: `maintenance/local-cleanup/20261110/unittest-restructure-final.log`. Không có failure/error mới trong lượt đó; chưa thể tuyên bố toàn bộ suite đạt.

Đối chiếu nội dung 18 launcher với bản Git trước thay đổi: sau khi chuẩn hóa đúng phép thay đường dẫn/root và xuống dòng, toàn bộ nội dung trùng khớp. `git diff -- src configs` rỗng. Helper dọn dẹp chỉ parse syntax, không thực hiện lại thao tác xóa trong đợt tái cấu trúc.
