# Code/source manifests

Các JSON ở đây là danh sách đường dẫn và SHA-256 dùng để kiểm tra code của từng phiên bản hoặc source RDF được cung cấp. Chúng không phải cấu hình inference, weights hay kết quả reconstruction.

- `pipeline-v22-code.json`, `pipeline-v23-code.json`: code theo phiên bản pipeline.
- `head24-code.json`, `head245-code.json`, `head26-code.json`: code và evidence của các nhánh head cũ.
- `hair-on-head-code.json`: các file thuộc tính năng hair-on-head.
- `head24-rdf-source.json`: file nguồn bên trong RDF repo; path của entry tính từ RDF repo, không tính từ configs/manifests.

Với manifest code còn lại, đường dẫn entry tính từ **repository root**. Di chuyển manifest không đổi gốc tính đường dẫn của entry. File evidence ở maintenance/verification có thể là tài liệu local của các bản nghiên cứu cũ, không phải dependency inference của assembly hiện tại.

Thay đổi path/hash code do tái cấu trúc được ghi trong docs/architecture/manifest-relocations-20261110.json và legacy-evidence-relocations-20261110.json. Không đổi model/asset pin để làm preflight vượt qua. Một số thiếu sót manifest đã tồn tại trước migration được giữ nguyên và báo trong kết quả test, không tạo file giả để che lỗi.
