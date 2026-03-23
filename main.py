import cv2
import os
import sys
import numpy as np
from datetime import datetime
from PIL import Image
import pickle
from insightface.app import FaceAnalysis

# ── Cấu hình ─────────────────────────────────────────
ENCODING_FILE    = "encodings.pkl"
NGUONG_COSINE    = 0.5   # khoảng cách cosine tối đa → nhỏ hơn = giống hơn
NGUONG_DETECT    = 0.5   # độ tin cậy detect tối thiểu
DEM_ON_DINH      = 3     # detect liên tiếp N lần mới hiện khung → chống chập chờn
DETECT_MOI       = 3     # detect 1 lần mỗi N frame → giảm tải GPU
CANH_BAO_MOI     = 3     # lưu ảnh người lạ tối đa 1 lần mỗi N giây

# ── Khởi tạo InsightFace ─────────────────────────────
def khoi_tao_model():
    print("Dang khoi tao InsightFace...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model san sang!\n")
    return app

# ── Tính cosine distance ──────────────────────────────
def cosine_distance(a, b):
    # Giá trị 0 = giống hệt, 2 = hoàn toàn khác
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ── Nhận diện danh tính ──────────────────────────────
def nhan_dien(embedding, known_embeddings, known_names):
    if not known_embeddings:
        return "NGUOI LA", 0.0

    distances = [cosine_distance(embedding, e) for e in known_embeddings]
    idx_min   = np.argmin(distances)
    dist_min  = distances[idx_min]

    if dist_min < NGUONG_COSINE:
        do_chinh_xac = (1 - dist_min) * 100
        return known_names[idx_min], do_chinh_xac

    return "NGUOI LA", 0.0

# ── Load hoặc cập nhật encodings ─────────────────────
def load_known_faces(app, folder="known_faces"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return [], []

    # Load pkl cũ nếu có
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, "rb") as f:
            data = pickle.load(f)
        known_embeddings = data["embeddings"]
        known_names      = data["names"]
        da_xu_ly         = set(data.get("da_xu_ly", []))
        print(f"Da load {len(known_embeddings)} embedding cu "
              f"cua {len(set(known_names))} nguoi.")
    else:
        known_embeddings = []
        known_names      = []
        da_xu_ly         = set()
        print("Chua co encodings — se tao moi...")

    # Chỉ xử lý ảnh MỚI chưa có trong pkl
    anh_moi = 0
    for ten in sorted(os.listdir(folder)):
        duong_dan = os.path.join(folder, ten)
        if not os.path.isdir(duong_dan):
            continue
        for file in sorted(os.listdir(duong_dan)):
            if not file.endswith((".jpg", ".png")):
                continue
            path = os.path.join(duong_dan, file)
            if path in da_xu_ly:
                continue  # bỏ qua ảnh đã xử lý

            try:
                anh   = np.array(Image.open(path).convert("RGB"))
                faces = app.get(anh)
                faces = [f for f in faces if f.det_score >= NGUONG_DETECT]

                if not faces:
                    print(f"  Khong thay mat: {file}")
                    da_xu_ly.add(path)
                    continue

                # Lấy khuôn mặt có det_score cao nhất
                face = max(faces, key=lambda f: f.det_score)
                known_embeddings.append(face.normed_embedding)
                known_names.append(ten)
                da_xu_ly.add(path)
                anh_moi += 1
                print(f"  [MOI] {ten}/{file} | score: {face.det_score:.2f}")

            except Exception as e:
                print(f"  Loi {file}: {e}")

    # Lưu pkl gộp cũ + mới
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({
            "embeddings": known_embeddings,
            "names":      known_names,
            "da_xu_ly":   da_xu_ly
        }, f)

    if anh_moi > 0:
        print(f"\nDa xu ly them {anh_moi} anh moi.")
    else:
        print("Khong co anh moi — dung luon embeddings cu.")

    print(f"Tong: {len(known_embeddings)} embedding "
          f"cua {len(set(known_names))} nguoi\n")
    return known_embeddings, known_names

# ── Xóa cache ────────────────────────────────────────
def xoa_cache():
    if os.path.exists(ENCODING_FILE):
        os.remove(ENCODING_FILE)
        print("Da xoa cache — se xu ly lai tat ca anh lan sau.")
    else:
        print("Chua co cache.")

# ── Lưu ảnh người lạ ────────────────────────────────
def luu_nguoi_la(frame):
    os.makedirs("unknown_log", exist_ok=True)
    ten_file = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    cv2.imwrite(f"unknown_log/{ten_file}", frame)
    print(f"[CANH BAO] Nguoi la! Luu: unknown_log/{ten_file}")

# ── Chương trình chính ───────────────────────────────
def main():
    print("=" * 50)
    print("  He thong nhan dien nguoi la — InsightFace")
    print("=" * 50)
    print("Lenh:")
    print("  python main.py           → chay binh thuong")
    print("  python main.py --update  → them anh moi")
    print("  python main.py --reset   → xu ly lai tu dau\n")

    # Xử lý tham số
    if "--reset" in sys.argv:
        xoa_cache()

    # Khởi tạo model
    app = khoi_tao_model()

    # Load embeddings
    known_embeddings, known_names = load_known_faces(app)

    # Nếu chỉ update thì dừng lại
    if "--update" in sys.argv:
        print("Cap nhat xong! Chay 'python main.py' de bat dau.")
        return

    if not known_embeddings:
        print("Chua co khuon mat! Hay chay capture_faces.py truoc.")
        return

    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Bat dau nhan dien — Nhan Q de thoat\n")

    thoi_gian_canh_bao = 0
    dem_frame          = 0
    bo_dem_on_dinh     = {}  # đếm số lần detect liên tiếp
    ket_qua_hien_tai   = []  # kết quả ổn định để vẽ lên mọi frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc frame!")
            break

        dem_frame += 1

        # Chỉ detect mỗi DETECT_MOI frame — tiết kiệm tài nguyên
        if dem_frame % DETECT_MOI == 0:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = app.get(rgb)
            faces = [f for f in faces if f.det_score >= NGUONG_DETECT]

            ket_qua_moi = []
            co_nguoi_la = False

            for face in faces:
                box        = face.bbox.astype(int)
                ten, do_cx = nhan_dien(
                    face.normed_embedding,
                    known_embeddings,
                    known_names
                )

                mau = (0, 255, 0) if ten != "NGUOI LA" else (0, 0, 255)
                if ten == "NGUOI LA":
                    co_nguoi_la = True

                ket_qua_moi.append((box, ten, do_cx, mau))

            # Smoothing — chỉ hiện khung khi detect ổn định
            if not ket_qua_moi:
                bo_dem_on_dinh   = {}
                ket_qua_hien_tai = []
            else:
                for item in ket_qua_moi:
                    key = item[1]
                    bo_dem_on_dinh[key] = bo_dem_on_dinh.get(key, 0) + 1

                ket_qua_hien_tai = [
                    item for item in ket_qua_moi
                    if bo_dem_on_dinh.get(item[1], 0) >= DEM_ON_DINH
                ]

            # Lưu ảnh người lạ — tối đa 1 lần mỗi 3 giây
            now = datetime.now().timestamp()
            if co_nguoi_la and (now - thoi_gian_canh_bao) > CANH_BAO_MOI:
                luu_nguoi_la(frame)
                thoi_gian_canh_bao = now

        # Vẽ kết quả lên mọi frame
        for (box, ten, do_cx, mau) in ket_qua_hien_tai:
            # Khung khuôn mặt
            cv2.rectangle(frame,
                          (box[0], box[1]), (box[2], box[3]),
                          mau, 2)
            # Nền tên
            cv2.rectangle(frame,
                          (box[0], box[3] - 30), (box[2], box[3]),
                          mau, cv2.FILLED)
            # Tên hiển thị
            nhan = ten if ten == "NGUOI LA" else f"{ten} ({do_cx:.0f}%)"
            cv2.putText(frame, nhan,
                        (box[0] + 4, box[3] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

        # Trạng thái hệ thống
        cv2.putText(frame,
                    f"Dang theo doi — {len(ket_qua_hien_tai)} khuon mat",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("He thong nhan dien nguoi la", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Da thoat.")

if __name__ == "__main__":
    main()