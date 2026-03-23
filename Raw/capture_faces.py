import cv2
import os
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis

# ── Khởi tạo InsightFace ─────────────────────────────
print("Dang khoi tao InsightFace...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("San sang!\n")

# ── Nhập thông tin ───────────────────────────────────
ten_nguoi = input("Nhap ten nguoi (khong dau, khong cach): ").strip()
folder    = f"known_faces/{ten_nguoi}"
os.makedirs(folder, exist_ok=True)

# Đếm ảnh đã có — để đánh số tiếp, không ghi đè
so_anh_cu = len([
    f for f in os.listdir(folder)
    if f.endswith((".jpg", ".png"))
])
print(f"Folder '{ten_nguoi}' hien co {so_anh_cu} anh.")

so_anh_them = int(input("Chup them bao nhieu anh? (goi y 15): ") or 15)

# ── Mở camera ───────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Gợi ý góc chụp — đa dạng để nhận diện tốt hơn
goi_y = [
    "Nhin THANG vao camera",
    "Nghieng MAT TRAI ~20 do",
    "Nghieng MAT PHAI ~20 do",
    "Cui dau xuong nhe",
    "Ngong dau len nhe",
    "Cuoi tu nhien",
    "Mat nghiem",
    "Xa camera ~1.5m",
    "Gan camera ~40cm",
    "Che nhe 1/4 mat trai",
    "Che nhe 1/4 mat phai",
    "Anh sang yeu (tat bot den)",
    "Nghieng dau sang trai",
    "Nghieng dau sang phai",
    "Tu nhien nhin camera",
]

dem  = 0
print(f"\nNhan S de chup, Q de thoat")
print(f"InsightFace kiem tra khuon mat truoc khi luu!\n")

while dem < so_anh_them:
    ret, frame = cap.read()
    if not ret:
        break

    hien_thi = frame.copy()

    # Detect khuôn mặt realtime — hiển thị khung để người dùng căn chỉnh
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)

    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(hien_thi,
                      (box[0], box[1]), (box[2], box[3]),
                      (0, 255, 0), 2)
        cv2.putText(hien_thi,
                    f"score: {face.det_score:.2f}",
                    (box[0], box[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Thông tin hiển thị
    cv2.putText(hien_thi, f"Nguoi: {ten_nguoi}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(hien_thi, f"Da chup: {dem}/{so_anh_them}  |  Nhan S",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Gợi ý góc chụp hiện tại
    if dem < len(goi_y):
        cv2.putText(hien_thi, f"Goi y: {goi_y[dem]}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2)

    # Trạng thái detect — báo ngay nếu không thấy mặt
    if faces:
        cv2.putText(hien_thi, f"Thay {len(faces)} khuon mat",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2)
    else:
        cv2.putText(hien_thi, "KHONG THAY KHUON MAT!",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)

    cv2.imshow("Chup anh nguoi quen", hien_thi)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if not faces:
            print("  Khong thay khuon mat! Hay dieu chinh lai vi tri.")
            continue

        # Lưu bằng PIL — tránh lỗi encoding Windows
        so_thu_tu = so_anh_cu + dem + 1
        path      = f"{folder}/{ten_nguoi}_{so_thu_tu}.jpg"
        anh_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(anh_rgb).save(path, format="JPEG", quality=95)

        dem += 1
        print(f"  Luu ({dem}/{so_anh_them}): {path}"
              f"  |  score: {faces[0].det_score:.2f}")

        if dem >= so_anh_them:
            print(f"\nXong! '{ten_nguoi}' gio co {so_anh_cu + dem} anh.")
            break

    elif key == ord('q'):
        print("Da thoat.")
        break

cap.release()
cv2.destroyAllWindows()
print("\nSau khi chup xong tat ca thanh vien:")
print("  → Chay 'python main.py --update' de tao encodings")