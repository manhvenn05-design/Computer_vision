# TÓM TẮT SẢN PHẨM
## Hệ Thống Phát Hiện và Cảnh Báo Người Lạ Vào Nhà

---

## 1. Tổng quan

Sản phẩm là ứng dụng desktop chạy trên Windows, sử dụng camera để giám sát và tự động phát hiện người lạ trong khu vực nhà ở. Khi phát hiện người không có trong danh sách người quen, hệ thống lập tức cảnh báo bằng âm thanh, hiển thị popup và lưu ảnh bằng chứng kèm thời gian.

---

## 2. Vấn đề giải quyết

Các hệ thống camera thông thường chỉ ghi hình mà không có khả năng phân biệt người quen và người lạ. Người dùng phải xem lại toàn bộ footage thủ công, tốn thời gian và dễ bỏ sót. Sản phẩm này giải quyết vấn đề đó bằng cách tự động nhận diện và cảnh báo ngay lập tức.

---

## 3. Công nghệ cốt lõi

Hệ thống sử dụng **InsightFace** — framework nhận diện khuôn mặt được dùng trong các sản phẩm thương mại thực tế, gồm 2 tầng:

**Tầng 1 — SCRFD:** Phát hiện khuôn mặt trong frame camera. Ưu điểm so với HOG truyền thống là nhận diện được khuôn mặt nghiêng, ánh sáng yếu và che khuất một phần.

**Tầng 2 — ArcFace:** Chuyển khuôn mặt thành vector 512 chiều gọi là "face embedding" — giống như dấu vân tay số duy nhất cho mỗi người. Khi có người xuất hiện trước camera, hệ thống tính khoảng cách cosine giữa embedding hiện tại với toàn bộ embedding trong database. Nếu khoảng cách nhỏ hơn ngưỡng 0.45 thì đây là người quen, ngược lại là người lạ.

---

## 4. Tính năng chính

**Nhận diện realtime:** Camera chạy 30fps, detect mỗi 3 frame để cân bằng tốc độ và tài nguyên. Cơ chế smoothing yêu cầu detect liên tiếp 2 lần mới hiển thị kết quả, tránh chập chờn.

**Quản lý người quen:** Chụp ảnh trực tiếp trong app hoặc chọn ảnh từ máy tính. Embeddings được cache vào file pkl — thêm người mới không cần xử lý lại toàn bộ database.

**Cảnh báo đa kênh:** Khi phát hiện người lạ, hệ thống đồng thời phát âm thanh beep 3 lần, hiển thị popup đỏ tự đóng sau 4 giây, ghi log có màu sắc phân biệt và lưu ảnh bằng chứng kèm bounding box + timestamp.

**Kho ảnh bằng chứng:** Ảnh người lạ lưu có khung nhận diện và thời gian. Giao diện gallery hỗ trợ lọc theo ngày, chọn xóa từng ảnh hoặc xóa theo nhóm.

**Xuất log:** Toàn bộ lịch sử cảnh báo có thể xuất ra file .txt để lưu trữ hoặc báo cáo.

---

## 5. Giao diện

Giao diện được xây dựng bằng Tkinter theo phong cách Material Design — nền trắng, màu sắc Google, font Segoe UI hỗ trợ tiếng Việt đầy đủ. Cửa sổ chính kích thước 1200×700px gồm:
- Bên trái: Camera feed 640×480px + 4 thống kê số liệu realtime
- Bên phải: Trạng thái hệ thống + 8 nút chức năng + Log cảnh báo màu sắc

---

## 6. Kết quả đạt được

- Nhận diện chính xác người quen trong điều kiện bình thường
- Phát hiện người lạ và cảnh báo trong vòng dưới 1 giây
- Hỗ trợ nhiều người quen cùng lúc trong database
- Giao diện hoàn toàn tiếng Việt, dễ sử dụng

---

## 7. Hạn chế

- Hiệu suất phụ thuộc vào chất lượng và số lượng ảnh đăng ký
- Chưa xử lý tốt trường hợp che mặt hoàn toàn
- Chỉ hỗ trợ 1 camera tại một thời điểm
- Lưu trữ bằng file pkl — chưa phù hợp cho quy mô lớn

---

## 8. Hướng phát triển

Tích hợp cơ sở dữ liệu SQLite để lưu lịch sử cảnh báo lâu dài. Hỗ trợ nhiều camera đồng thời. Gửi thông báo qua Telegram hoặc Email. Triển khai trên Raspberry Pi cho hệ thống nhúng chi phí thấp.

---

*Nhóm 6 — Môn Thị Giác Máy Tính — Trường Đại Học Vinh*
*Giáo viên hướng dẫn: Nguyễn Thị Minh Tâm*
