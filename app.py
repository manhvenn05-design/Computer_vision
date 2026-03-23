import cv2
import tkinter as tk
from tkinter import ttk, scrolledtext, font
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk
import pickle
import os
from insightface.app import FaceAnalysis

# ── Cấu hình ─────────────────────────────────────────
ENCODING_FILE = "encodings.pkl"
NGUONG_COSINE = 0.45
NGUONG_DETECT = 0.5
DEM_ON_DINH   = 2
DETECT_MOI    = 3
CANH_BAO_MOI  = 3

# ── Màu sắc ──────────────────────────────────────────
BG_MAIN    = "#F0F4F8"
BG_CARD    = "#FFFFFF"
BG_HEADER  = "#1A73E8"
CLR_GREEN  = "#34A853"
CLR_RED    = "#EA4335"
CLR_BLUE   = "#1A73E8"
CLR_ORANGE = "#FBBC04"
CLR_GRAY   = "#5F6368"
CLR_LIGHT  = "#E8F0FE"
CLR_TEXT   = "#202124"
CLR_SUB    = "#5F6368"

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def nhan_dien(embedding, known_embeddings, known_names):
    if not known_embeddings:
        return "NGUOI LA", 0.0
    distances = [cosine_distance(embedding, e) for e in known_embeddings]
    idx_min   = np.argmin(distances)
    dist_min  = distances[idx_min]
    if dist_min < NGUONG_COSINE:
        return known_names[idx_min], (1 - dist_min) * 100
    return "NGUOI LA", 0.0

def load_encodings():
    if not os.path.exists(ENCODING_FILE):
        return [], []
    with open(ENCODING_FILE, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["names"]

def luu_nguoi_la(frame, ket_qua):
    os.makedirs("unknown_log", exist_ok=True)
    ten_file = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    anh_luu = frame.copy()

    # Vẽ khung xanh cho người quen, đỏ cho người lạ
    for (box, ten, do_cx, mau) in ket_qua:
        cv2.rectangle(anh_luu, (box[0], box[1]), (box[2], box[3]), mau, 2)
        cv2.rectangle(anh_luu, (box[0], box[3] - 32), (box[2], box[3]), mau, cv2.FILLED)
        nhan = ten if ten == "NGUOI LA" else f"{ten} ({do_cx:.0f}%)"
        cv2.putText(anh_luu, nhan, (box[0] + 4, box[3] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Vẽ timestamp góc trên trái
    gio = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
    cv2.rectangle(anh_luu, (0, 0), (280, 28), (0, 0, 0), cv2.FILLED)
    cv2.putText(anh_luu, gio, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 1)

    cv2.imwrite(f"unknown_log/{ten_file}", anh_luu)
    return ten_file


class App:
    def __init__(self, root):
        self.root  = root
        self.root.title("Hệ Thống Nhận Diện Người Lạ")
        self.root.configure(bg=BG_MAIN)
        self.root.resizable(False, False)

        w, h = 1200, 700
        sw   = self.root.winfo_screenwidth()
        sh   = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self.dang_chay          = False
        self.cap                = None
        self.dem_frame          = 0
        self.thoi_gian_canh_bao = 0
        self.bo_dem_on_dinh     = {}
        self.ket_qua_hien_tai   = []
        self.tong_canh_bao      = 0
        self.app_insight        = None
        self.known_embeddings   = []
        self.known_names        = []
        self._popup_dang_hien   = False

        self._build_ui()
        self._load_model_thread()

    def _card(self, parent, **kwargs):
        return tk.Frame(parent, bg=BG_CARD, relief="flat",
                        highlightthickness=1,
                        highlightbackground="#DADCE0", **kwargs)

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG_HEADER, height=52)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="  🏠  Hệ Thống Nhận Diện Người Lạ",
                 bg=BG_HEADER, fg="white",
                 font=("Segoe UI", 14, "bold")).pack(side="left", pady=10)

        self.label_gio = tk.Label(header, text="",
                                   bg=BG_HEADER, fg="#E8F0FE",
                                   font=("Segoe UI", 10))
        self.label_gio.pack(side="right", padx=16)
        self._cap_nhat_gio()

        # Body
        body = tk.Frame(self.root, bg=BG_MAIN)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # Cột trái: Camera
        col_trai = tk.Frame(body, bg=BG_MAIN)
        col_trai.pack(side="left", fill="both", expand=True)

        card_cam = self._card(col_trai)
        card_cam.pack(fill="both", expand=True, pady=(0, 8))

        tk.Label(card_cam, text="📷  Camera Giám Sát",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Frame(card_cam, bg="#DADCE0", height=1).pack(fill="x", padx=12)

        frame_cam_inner = tk.Frame(card_cam, bg="#000000", width=640, height=480)
        frame_cam_inner.pack(padx=12, pady=8)
        frame_cam_inner.pack_propagate(False)

        self.label_camera = tk.Label(frame_cam_inner, bg="#000000")
        self.label_camera.pack(fill="both", expand=True)

        # Thống kê
        card_stat = self._card(col_trai)
        card_stat.pack(fill="x")

        frame_stat = tk.Frame(card_stat, bg=BG_CARD)
        frame_stat.pack(fill="x", padx=12, pady=8)

        stats = [
            ("Khuôn mặt", "0",   CLR_BLUE,   "label_so_mat"),
            ("Người quen", "0",  CLR_GREEN,  "label_so_quen"),
            ("Người lạ",   "0",  CLR_RED,    "label_so_la"),
            ("DB (ảnh)",   "...", CLR_ORANGE, "label_db"),
        ]
        for i, (nhan, val, mau, attr) in enumerate(stats):
            f = tk.Frame(frame_stat, bg=BG_CARD)
            f.pack(side="left", expand=True)
            lbl_val = tk.Label(f, text=val, bg=BG_CARD, fg=mau,
                                font=("Segoe UI", 20, "bold"))
            lbl_val.pack()
            tk.Label(f, text=nhan, bg=BG_CARD, fg=CLR_SUB,
                     font=("Segoe UI", 8)).pack()
            setattr(self, attr, lbl_val)
            if i < len(stats) - 1:
                tk.Frame(frame_stat, bg="#DADCE0",
                         width=1).pack(side="left", fill="y", padx=8)

        # Cột phải: Điều khiển
        col_phai = tk.Frame(body, bg=BG_MAIN, width=300)
        col_phai.pack(side="left", fill="y", padx=(10, 0))
        col_phai.pack_propagate(False)

        # Card trạng thái
        card_tt = self._card(col_phai)
        card_tt.pack(fill="x", pady=(0, 8))

        tk.Label(card_tt, text="⚡  Trạng Thái Hệ Thống",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Frame(card_tt, bg="#DADCE0", height=1).pack(fill="x", padx=12)

        self.label_trang_thai = tk.Label(
            card_tt, text="Đang khởi tạo...",
            bg=CLR_LIGHT, fg=CLR_BLUE,
            font=("Segoe UI", 9),
            relief="flat", padx=8, pady=6,
            wraplength=260, justify="left"
        )
        self.label_trang_thai.pack(fill="x", padx=12, pady=8)

        # Card điều khiển
        card_dk = self._card(col_phai)
        card_dk.pack(fill="x", pady=(0, 8))

        tk.Label(card_dk, text="🎮  Điều Khiển",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Frame(card_dk, bg="#DADCE0", height=1).pack(fill="x", padx=12)

        frame_btn = tk.Frame(card_dk, bg=BG_CARD)
        frame_btn.pack(fill="x", padx=12, pady=8)

        self.btn_bat_dau = tk.Button(
            frame_btn, text="▶  Bắt Đầu",
            bg=CLR_GREEN, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat", pady=7, cursor="hand2",
            state="disabled", bd=0,
            activebackground="#2D9142",
            command=self.bat_dau)
        self.btn_bat_dau.pack(fill="x", pady=2)

        self.btn_dung = tk.Button(
            frame_btn, text="■  Dừng Lại",
            bg=CLR_RED, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat", pady=7, cursor="hand2",
            state="disabled", bd=0,
            activebackground="#C5221F",
            command=self.dung_lai)
        self.btn_dung.pack(fill="x", pady=2)

        for text, bg, active, cmd in [
            ("📷  Chụp Ảnh Người Quen", CLR_BLUE,    "#1557B0", self.chup_anh_nguoi_quen),
            ("🖼  Thêm Ảnh Từ Máy",     "#7B1FA2",   "#6A1B9A", self.them_anh_tu_may),
            ("👥  Quản Lý Người Quen",  "#37474F",   "#263238", self.quan_ly_nguoi_quen),
            ("📂  Xem Ảnh Người Lạ",   "#FF6D00",   "#E65100", self.xem_anh_nguoi_la),
            ("📤  Xuất Log Ra File",    "#00897B",   "#00695C", self.xuat_log),
        ]:
            tk.Button(frame_btn, text=text, bg=bg, fg="white",
                      font=("Segoe UI", 9, "bold"),
                      relief="flat", pady=6, cursor="hand2",
                      bd=0, activebackground=active,
                      command=cmd).pack(fill="x", pady=2)

        tk.Button(frame_btn, text="🗑  Xóa Log",
                  bg="#F1F3F4", fg=CLR_GRAY,
                  font=("Segoe UI", 9),
                  relief="flat", pady=6, cursor="hand2",
                  bd=0, command=self.xoa_log).pack(fill="x", pady=2)

        tk.Button(frame_btn, text="ℹ  Về Chúng Tôi",
                  bg="#F1F3F4", fg=CLR_GRAY,
                  font=("Segoe UI", 9),
                  relief="flat", pady=6, cursor="hand2",
                  bd=0, command=self.ve_chung_toi).pack(fill="x", pady=(8, 2))

        # Card log
        card_log = self._card(col_phai)
        card_log.pack(fill="both", expand=True)

        tk.Label(card_log, text="🔔  Log Cảnh Báo",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Frame(card_log, bg="#DADCE0", height=1).pack(fill="x", padx=12)

        self.log_text = scrolledtext.ScrolledText(
            card_log, width=30, height=12,
            bg="#FAFAFA", fg=CLR_TEXT,
            font=("Consolas", 8),
            relief="flat", state="disabled",
            padx=6, pady=4)
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _load_model_thread(self):
        self._ghi_log("Đang khởi tạo InsightFace...")

        def _load():
            try:
                self.app_insight = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                self.app_insight.prepare(ctx_id=0, det_size=(640, 640))
                self.known_embeddings, self.known_names = load_encodings()
                so_nguoi = len(set(self.known_names))
                so_anh   = len(self.known_embeddings)

                def _update():
                    self.label_db.config(text=str(so_anh))
                    self.label_so_quen.config(text=str(so_nguoi))
                    self.label_trang_thai.config(
                        text=f"Sẵn sàng!\n{so_anh} ảnh / {so_nguoi} người quen",
                        bg="#E6F4EA", fg=CLR_GREEN)
                    self.btn_bat_dau.config(state="normal")
                    self._ghi_log(f"[OK] Model sẵn sàng")
                    self._ghi_log(f"[OK] {so_anh} ảnh / {so_nguoi} người quen")

                self.root.after(0, _update)
            except Exception as e:
                self.root.after(0, lambda: self._ghi_log(f"[LOI] {e}"))

        threading.Thread(target=_load, daemon=True).start()

    def bat_dau(self):
        if not self.known_embeddings:
            from tkinter import messagebox
            messagebox.showwarning(
                "Chưa có dữ liệu",
                "Chưa có khuôn mặt nào trong DB!\n\n"
                "Hãy thêm người quen trước bằng cách:\n"
                "• Nhấn '📷 Chụp Ảnh Người Quen'\n"
                "• Hoặc '🖼 Thêm Ảnh Từ Máy'",
                parent=self.root)
            self._ghi_log("[!] Chưa có dữ liệu — hãy thêm người quen trước!")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._ghi_log("[!] Không mở được camera!")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.dang_chay          = True
        self.dem_frame          = 0
        self.thoi_gian_canh_bao = 0
        self.bo_dem_on_dinh     = {}
        self.ket_qua_hien_tai   = []

        self.btn_bat_dau.config(state="disabled")
        self.btn_dung.config(state="normal")
        self.label_trang_thai.config(text="Đang theo dõi...", bg=CLR_LIGHT, fg=CLR_BLUE)
        self._ghi_log("[>>] Bắt đầu theo dõi!")
        self._cap_nhat_camera()

    def dung_lai(self):
        self.dang_chay = False
        if self.cap:
            self.cap.release()
        self.label_camera.config(image="", bg="#000000")
        self.btn_bat_dau.config(state="normal")
        self.btn_dung.config(state="disabled")
        self.label_trang_thai.config(text="Đã dừng.", bg="#FCE8E6", fg=CLR_RED)
        self.label_so_mat.config(text="0")
        self._ghi_log("[■] Đã dừng theo dõi.")

    def chup_anh_nguoi_quen(self):
        tam_dung = self.dang_chay
        if self.dang_chay:
            self.dang_chay = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.label_camera.config(image="", bg="#000000")
            self._ghi_log("[~] Tạm dừng camera chính.")

        win = tk.Toplevel(self.root)
        win.title("Chụp Ảnh Người Quen")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)

        tk.Label(win, text="📷  Chụp Ảnh Người Quen",
                 bg=BG_HEADER, fg="white",
                 font=("Segoe UI", 12, "bold")).pack(fill="x", ipady=8)

        frame_ten = tk.Frame(win, bg=BG_MAIN)
        frame_ten.pack(fill="x", padx=12, pady=6)
        tk.Label(frame_ten, text="Tên người:",
                 bg=BG_MAIN, fg=CLR_TEXT,
                 font=("Segoe UI", 10)).pack(side="left")
        entry_ten = tk.Entry(frame_ten, font=("Segoe UI", 10), width=20)
        entry_ten.pack(side="left", padx=8)
        entry_ten.focus()

        frame_cam = tk.Frame(win, bg="#000000", width=640, height=420)
        frame_cam.pack(padx=12, pady=4)
        frame_cam.pack_propagate(False)
        label_preview = tk.Label(frame_cam, bg="#000000")
        label_preview.pack(fill="both", expand=True)

        label_info = tk.Label(win,
                               text="Nhập tên → Bắt Đầu → Nhấn S để chụp",
                               bg=BG_MAIN, fg=CLR_BLUE,
                               font=("Segoe UI", 9, "bold"))
        label_info.pack(pady=2)

        state = {"chup": False, "cap": None, "dem": 0,
                 "so_cu": 0, "ten": "", "frame": None}

        def bat_dau():
            ten = entry_ten.get().strip().replace(" ", "_")
            if not ten:
                label_info.config(text="Hãy nhập tên!", fg=CLR_RED)
                return
            state["ten"]  = ten
            state["dem"]  = 0
            state["chup"] = True
            os.makedirs(f"known_faces/{ten}", exist_ok=True)
            state["so_cu"] = len([
                f for f in os.listdir(f"known_faces/{ten}")
                if f.endswith(".jpg")])
            state["cap"] = cv2.VideoCapture(0)
            state["cap"].set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            state["cap"].set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
            btn_bat_dau.config(state="disabled")
            btn_chup.config(state="normal")
            label_info.config(text="Nhấn S để chụp | 0 ảnh", fg=CLR_GREEN)
            _loop()

        def _loop():
            if not state["chup"] or not state["cap"]:
                return
            ret, frame = state["cap"].read()
            if ret:
                state["frame"] = frame
                img   = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 420))
                imgtk = ImageTk.PhotoImage(img)
                label_preview.imgtk = imgtk
                label_preview.config(image=imgtk)
            win.after(30, _loop)

        def chup():
            if not state["chup"] or state["frame"] is None:
                return
            so = state["so_cu"] + state["dem"] + 1
            p  = f"known_faces/{state['ten']}/{state['ten']}_{so}.jpg"
            Image.fromarray(
                cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB)
            ).save(p, format="JPEG", quality=95)
            state["dem"] += 1
            label_info.config(
                text=f"Đã chụp {state['dem']} ảnh | Nhấn S tiếp tục",
                fg=CLR_GREEN)

        def luu_dong():
            state["chup"] = False
            if state["cap"]:
                state["cap"].release()
            if state["dem"] > 0:
                self._ghi_log(f"[OK] Đã chụp {state['dem']} ảnh cho '{state['ten']}'")
                self._cap_nhat_encoding_silent()
            if tam_dung:
                self.root.after(800, self.bat_dau)
            win.destroy()

        def huy():
            state["chup"] = False
            if state["cap"]:
                state["cap"].release()
            if tam_dung:
                self.root.after(800, self.bat_dau)
            win.destroy()

        win.bind("<s>", lambda e: chup())
        win.bind("<S>", lambda e: chup())

        frame_btn = tk.Frame(win, bg=BG_MAIN)
        frame_btn.pack(fill="x", padx=12, pady=6)

        btn_bat_dau = tk.Button(frame_btn, text="▶  Bắt Đầu",
                                 bg=CLR_GREEN, fg="white",
                                 font=("Segoe UI", 9, "bold"),
                                 relief="flat", pady=5, cursor="hand2",
                                 bd=0, command=bat_dau)
        btn_bat_dau.pack(side="left", padx=4)

        btn_chup = tk.Button(frame_btn, text="📸  Chụp! (S)",
                              bg=CLR_ORANGE, fg="white",
                              font=("Segoe UI", 9, "bold"),
                              relief="flat", pady=5, cursor="hand2",
                              state="disabled", bd=0, command=chup)
        btn_chup.pack(side="left", padx=4)

        tk.Button(frame_btn, text="💾  Lưu & Đóng",
                  bg=CLR_BLUE, fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", pady=5, cursor="hand2",
                  bd=0, command=luu_dong).pack(side="left", padx=4)

        tk.Button(frame_btn, text="✕  Hủy",
                  bg="#F1F3F4", fg=CLR_GRAY,
                  font=("Segoe UI", 9),
                  relief="flat", pady=5, cursor="hand2",
                  bd=0, command=huy).pack(side="left", padx=4)

        win.protocol("WM_DELETE_WINDOW", huy)

    def them_anh_tu_may(self):
        from tkinter import filedialog, simpledialog
        ten = simpledialog.askstring(
            "Tên người",
            "Nhập tên người (không dấu, không cách):",
            parent=self.root)
        if not ten or not ten.strip():
            return
        ten = ten.strip().replace(" ", "_")

        files = filedialog.askopenfilenames(
            title=f"Chọn ảnh cho '{ten}'",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if not files:
            return

        folder = f"known_faces/{ten}"
        os.makedirs(folder, exist_ok=True)
        so_cu = len([f for f in os.listdir(folder)
                     if f.endswith((".jpg", ".png"))])
        dem = 0
        for i, src in enumerate(files):
            dst = f"{folder}/{ten}_{so_cu + i + 1}.jpg"
            try:
                Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)
                dem += 1
            except Exception as e:
                self._ghi_log(f"[!] Lỗi copy {src}: {e}")

        self._ghi_log(f"[OK] Đã thêm {dem} ảnh cho '{ten}'")

        from tkinter import messagebox
        messagebox.showinfo(
            "Thêm ảnh thành công!",
            f"Đã thêm {dem} ảnh cho '{ten}'\nĐang cập nhật encoding...",
            parent=self.root)
        self._cap_nhat_encoding_silent()

    def _cap_nhat_encoding_silent(self):
        self._ghi_log("[~] Đang cập nhật encoding...")

        def _run():
            try:
                folder = "known_faces"
                if not os.path.exists(folder):
                    return

                if os.path.exists(ENCODING_FILE):
                    with open(ENCODING_FILE, "rb") as f:
                        data = pickle.load(f)
                    known_embeddings = data["embeddings"]
                    known_names      = data["names"]
                    da_xu_ly         = set(data.get("da_xu_ly", []))
                else:
                    known_embeddings, known_names, da_xu_ly = [], [], set()

                anh_moi = 0
                for ten in os.listdir(folder):
                    dp = os.path.join(folder, ten)
                    if not os.path.isdir(dp):
                        continue
                    for file in sorted(os.listdir(dp)):
                        if not file.endswith((".jpg", ".png")):
                            continue
                        path = os.path.join(dp, file)
                        if path in da_xu_ly:
                            continue
                        try:
                            anh   = np.array(Image.open(path).convert("RGB"))
                            faces = self.app_insight.get(anh)
                            faces = [f for f in faces if f.det_score >= NGUONG_DETECT]
                            if not faces:
                                da_xu_ly.add(path)
                                continue
                            face = max(faces, key=lambda f: f.det_score)
                            known_embeddings.append(face.normed_embedding)
                            known_names.append(ten)
                            da_xu_ly.add(path)
                            anh_moi += 1
                        except Exception as e:
                            print(f"Lỗi {file}: {e}")

                with open(ENCODING_FILE, "wb") as f:
                    pickle.dump({"embeddings": known_embeddings,
                                 "names": known_names,
                                 "da_xu_ly": da_xu_ly}, f)

                self.known_embeddings = list(known_embeddings)
                self.known_names      = list(known_names)
                so_nguoi = len(set(self.known_names))
                so_anh   = len(self.known_embeddings)

                def _ui():
                    self.label_db.config(text=str(so_anh))
                    self.label_so_quen.config(text=str(so_nguoi))
                    self.label_trang_thai.config(
                        text=f"Cập nhật xong!\n{so_anh} ảnh / {so_nguoi} người",
                        bg="#E6F4EA", fg=CLR_GREEN)
                    self._ghi_log(
                        f"[OK] Cập nhật xong: +{anh_moi} ảnh mới | "
                        f"Tổng: {so_anh} ảnh / {so_nguoi} người")

                self.root.after(0, _ui)
            except Exception as e:
                self.root.after(0, lambda: self._ghi_log(f"[LOI] {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _cap_nhat_camera(self):
        if not self.dang_chay:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._cap_nhat_camera)
            return

        self.dem_frame += 1

        if self.dem_frame % DETECT_MOI == 0:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.app_insight.get(rgb)
            faces = [f for f in faces if f.det_score >= NGUONG_DETECT]

            ket_qua_moi = []
            co_nguoi_la = False

            for face in faces:
                box        = face.bbox.astype(int)
                ten, do_cx = nhan_dien(face.normed_embedding,
                                        self.known_embeddings,
                                        self.known_names)
                mau = (52, 168, 83) if ten != "NGUOI LA" else (234, 67, 53)
                if ten == "NGUOI LA":
                    co_nguoi_la = True
                ket_qua_moi.append((box, ten, do_cx, mau))

            if not ket_qua_moi:
                self.bo_dem_on_dinh   = {}
                self.ket_qua_hien_tai = []
            else:
                for item in ket_qua_moi:
                    key = item[1]
                    self.bo_dem_on_dinh[key] = self.bo_dem_on_dinh.get(key, 0) + 1
                self.ket_qua_hien_tai = [
                    item for item in ket_qua_moi
                    if self.bo_dem_on_dinh.get(item[1], 0) >= DEM_ON_DINH]

            now = datetime.now().timestamp()
            if co_nguoi_la and (now - self.thoi_gian_canh_bao) > CANH_BAO_MOI:
                ten_file = luu_nguoi_la(frame, ket_qua_moi)
                self.tong_canh_bao += 1
                self.thoi_gian_canh_bao = now
                gio = datetime.now().strftime("%H:%M:%S")
                self._ghi_log(f"[!!!] {gio} — NGƯỜI LẠ phát hiện! 📁 {ten_file}", "canh_bao")
                self.label_so_la.config(text=str(self.tong_canh_bao))
                threading.Thread(target=self._canh_bao_nguoi_la, daemon=True).start()

            self.label_so_mat.config(text=str(len(self.ket_qua_hien_tai)))

        for (box, ten, do_cx, mau) in self.ket_qua_hien_tai:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), mau, 2)
            cv2.rectangle(frame, (box[0], box[3] - 32), (box[2], box[3]), mau, cv2.FILLED)
            nhan = ten if ten == "NGUOI LA" else f"{ten} ({do_cx:.0f}%)"
            cv2.putText(frame, nhan, (box[0] + 4, box[3] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        w = self.label_camera.winfo_width()
        h = self.label_camera.winfo_height()
        if w < 10 or h < 10:
            w, h = 640, 480

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img       = Image.fromarray(frame_rgb).resize((w, h))
        imgtk     = ImageTk.PhotoImage(image=img)
        self.label_camera.imgtk = imgtk
        self.label_camera.config(image=imgtk)
        self.root.after(15, self._cap_nhat_camera)

    def _ghi_log(self, msg, loai="info"):
        if loai == "info":
            if msg.startswith("[!!!]"):
                loai = "canh_bao"
            elif msg.startswith("[OK]") or msg.startswith("[>>]"):
                loai = "ok"
            elif msg.startswith("[~]") or msg.startswith("[■]"):
                loai = "he_thong"

        mau = {
            "canh_bao": "#EA4335",
            "ok":       "#34A853",
            "he_thong": "#1A73E8",
            "info":     "#202124",
        }.get(loai, "#202124")

        tag = f"tag_{loai}"
        self.log_text.config(state="normal")
        gio = datetime.now().strftime("%H:%M:%S")
        self.log_text.tag_config(tag, foreground=mau)
        self.log_text.insert("end", f"{gio}  {msg}\n", tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _canh_bao_nguoi_la(self):
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 200)
        except:
            pass

        if not self._popup_dang_hien:
            self._popup_dang_hien = True
            self.root.after(0, self._hien_popup_canh_bao)

    def _hien_popup_canh_bao(self):
        popup = tk.Toplevel(self.root)
        popup.title("⚠ CẢNH BÁO")
        popup.configure(bg="#EA4335")
        popup.resizable(False, False)
        popup.attributes("-topmost", True)

        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        popup.geometry(f"360x200+{(sw-360)//2}+{(sh-200)//2}")

        tk.Label(popup, text="⚠",
                 bg="#EA4335", fg="white",
                 font=("Segoe UI", 36)).pack(pady=(12, 4))
        tk.Label(popup, text="PHÁT HIỆN NGƯỜI LẠ!",
                 bg="#EA4335", fg="white",
                 font=("Segoe UI", 14, "bold")).pack()
        tk.Label(popup, text=f"Thời gian: {datetime.now().strftime('%H:%M:%S')}",
                 bg="#EA4335", fg="#FFCDD2",
                 font=("Segoe UI", 10)).pack(pady=4)

        label_dem = tk.Label(popup, text="Tự đóng sau 4 giây",
                              bg="#EA4335", fg="#FFCDD2",
                              font=("Segoe UI", 9))
        label_dem.pack()

        def dem_nguoc(n):
            if n <= 0:
                self._popup_dang_hien = False
                try:
                    popup.destroy()
                except:
                    pass
                return
            try:
                label_dem.config(text=f"Tự đóng sau {n} giây")
                popup.after(1000, lambda: dem_nguoc(n - 1))
            except:
                self._popup_dang_hien = False

        tk.Button(popup, text="✕  Đóng",
                  bg="#C62828", fg="white",
                  font=("Segoe UI", 9, "bold"),
                  relief="flat", pady=4, cursor="hand2", bd=0,
                  command=lambda: [
                      setattr(self, '_popup_dang_hien', False),
                      popup.destroy()
                  ]).pack(pady=6)

        popup.after(1000, lambda: dem_nguoc(3))

    def quan_ly_nguoi_quen(self):
        from tkinter import messagebox

        folder = "known_faces"
        if not os.path.exists(folder):
            self._ghi_log("[!] Chưa có người quen nào.")
            return

        nguoi_list = sorted([
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))])

        if not nguoi_list:
            self._ghi_log("[!] Danh sách người quen trống.")
            return

        win = tk.Toplevel(self.root)
        win.title("Quản Lý Người Quen")
        win.configure(bg=BG_MAIN)
        win.geometry("780x520")
        win.resizable(False, False)

        tk.Label(win, text="👥  Quản Lý Người Quen",
                 bg=BG_HEADER, fg="white",
                 font=("Segoe UI", 12, "bold")).pack(fill="x", ipady=8)

        body = tk.Frame(win, bg=BG_MAIN)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Cột trái: danh sách
        col_trai = tk.Frame(body, bg=BG_MAIN, width=280)
        col_trai.pack(side="left", fill="y")
        col_trai.pack_propagate(False)

        tk.Label(col_trai, text="Danh sách người quen:",
                 bg=BG_MAIN, fg=CLR_TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        frame_list = tk.Frame(col_trai, bg=BG_MAIN)
        frame_list.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(frame_list)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(frame_list,
                              font=("Segoe UI", 11),
                              selectmode="single",
                              bg=BG_CARD, fg=CLR_TEXT,
                              selectbackground=CLR_BLUE,
                              selectforeground="white",
                              relief="flat",
                              yscrollcommand=scrollbar.set,
                              highlightthickness=1,
                              highlightbackground="#DADCE0")
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        # Cột phải: xem trước ảnh
        col_phai = tk.Frame(body, bg=BG_MAIN)
        col_phai.pack(side="left", fill="both", expand=True, padx=(10, 0))

        tk.Label(col_phai, text="Xem trước ảnh:",
                 bg=BG_MAIN, fg=CLR_TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        frame_preview = tk.Frame(col_phai, bg="#F8F9FA",
                                  highlightthickness=1,
                                  highlightbackground="#DADCE0")
        frame_preview.pack(fill="both", expand=True)

        canvas_prev = tk.Canvas(frame_preview, bg="#F8F9FA", highlightthickness=0)
        sb_prev     = tk.Scrollbar(frame_preview, orient="vertical",
                                    command=canvas_prev.yview)
        frame_imgs  = tk.Frame(canvas_prev, bg="#F8F9FA")
        frame_imgs.bind("<Configure>", lambda e: canvas_prev.configure(
            scrollregion=canvas_prev.bbox("all")))
        canvas_prev.create_window((0, 0), window=frame_imgs, anchor="nw")
        canvas_prev.configure(yscrollcommand=sb_prev.set)
        canvas_prev.pack(side="left", fill="both", expand=True)
        sb_prev.pack(side="right", fill="y")

        label_ten_preview = tk.Label(col_phai, text="← Chọn tên để xem ảnh",
                                      bg=BG_MAIN, fg=CLR_SUB,
                                      font=("Segoe UI", 9))
        label_ten_preview.pack(pady=4)

        ds = list(nguoi_list)

        def refresh_list():
            listbox.delete(0, "end")
            ds.clear()
            ds.extend(sorted([
                d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))]))
            for ten in ds:
                so_anh = len([f for f in os.listdir(os.path.join(folder, ten))
                               if f.endswith((".jpg", ".png"))])
                listbox.insert("end", f"  👤  {ten}  ({so_anh} ảnh)")

        refresh_list()

        def hien_anh_preview(ten):
            for w in frame_imgs.winfo_children():
                w.destroy()
            anh_folder = os.path.join(folder, ten)
            anh_files  = sorted([
                f for f in os.listdir(anh_folder)
                if f.endswith((".jpg", ".png"))])[:12]
            COLS, THUMB = 3, 120
            for i, fn in enumerate(anh_files):
                try:
                    img   = Image.open(os.path.join(anh_folder, fn)).resize((THUMB, THUMB))
                    imgtk = ImageTk.PhotoImage(img)
                    lbl   = tk.Label(frame_imgs, image=imgtk, bg="#F8F9FA")
                    lbl.image = imgtk
                    lbl.grid(row=i // COLS, column=i % COLS, padx=3, pady=3)
                except:
                    pass
            so_anh = len([f for f in os.listdir(anh_folder)
                          if f.endswith((".jpg", ".png"))])
            label_ten_preview.config(
                text=f"👤  {ten}  —  {so_anh} ảnh trong DB",
                fg=CLR_BLUE, font=("Segoe UI", 9, "bold"))

        listbox.bind("<<ListboxSelect>>",
                     lambda e: hien_anh_preview(ds[listbox.curselection()[0]])
                     if listbox.curselection() else None)

        label_chon = tk.Label(win, text="Chưa chọn ai",
                               bg=BG_MAIN, fg=CLR_SUB,
                               font=("Segoe UI", 9))
        label_chon.pack(pady=2)

        def xoa_nguoi():
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("Chưa chọn", "Hãy chọn người muốn xóa!", parent=win)
                return
            ten = ds[sel[0]]
            so_anh_xoa = len([f for f in os.listdir(os.path.join(folder, ten))
                               if f.endswith(".jpg")])
            if not messagebox.askyesno(
                "Xác nhận xóa",
                f"Bạn có chắc muốn xóa '{ten}'?\n"
                f"Tất cả {so_anh_xoa} ảnh sẽ bị xóa!",
                parent=win):
                return

            import shutil
            shutil.rmtree(os.path.join(folder, ten))
            self._ghi_log(f"[OK] Đã xóa '{ten}' khỏi DB.")

            if os.path.exists(ENCODING_FILE):
                with open(ENCODING_FILE, "rb") as f:
                    data = pickle.load(f)
                moi_emb, moi_name, moi_daxl = [], [], set()
                for emb, name, path_xl in zip(
                        data["embeddings"], data["names"],
                        data.get("da_xu_ly", [])):
                    if name != ten:
                        moi_emb.append(emb)
                        moi_name.append(name)
                        moi_daxl.add(path_xl)
                with open(ENCODING_FILE, "wb") as f:
                    pickle.dump({"embeddings": moi_emb, "names": moi_name,
                                 "da_xu_ly": moi_daxl}, f)
                self.known_embeddings = list(moi_emb)
                self.known_names      = list(moi_name)
                so_nguoi = len(set(self.known_names))
                so_anh   = len(self.known_embeddings)
                self.label_db.config(text=str(so_anh))
                self.label_so_quen.config(text=str(so_nguoi))
                self.label_trang_thai.config(
                    text=f"Đã xóa '{ten}'\nCòn lại: {so_anh} ảnh / {so_nguoi} người",
                    bg="#E6F4EA", fg=CLR_GREEN)
                self._ghi_log(f"[OK] Còn lại: {so_anh} ảnh / {so_nguoi} người")

            for w in frame_imgs.winfo_children():
                w.destroy()
            label_ten_preview.config(text="Đã xóa thành công!", fg=CLR_GREEN)
            label_chon.config(text=f"Đã xóa '{ten}'", fg=CLR_GREEN)
            refresh_list()

        tk.Button(win, text="🗑  Xóa Người Này",
                  bg=CLR_RED, fg="white",
                  font=("Segoe UI", 10, "bold"),
                  relief="flat", pady=7, cursor="hand2",
                  bd=0, command=xoa_nguoi).pack(fill="x", padx=10, pady=(2, 2))

        tk.Button(win, text="✕  Đóng",
                  bg="#F1F3F4", fg=CLR_GRAY,
                  font=("Segoe UI", 9),
                  relief="flat", pady=6, cursor="hand2",
                  bd=0, command=win.destroy).pack(fill="x", padx=10, pady=(0, 8))

    def xem_anh_nguoi_la(self):
        folder = "unknown_log"
        if not os.path.exists(folder) or not any(
                f.endswith(".jpg") for f in os.listdir(folder)):
            self._ghi_log("[!] Chưa có ảnh người lạ nào.")
            return

        win = tk.Toplevel(self.root)
        win.title("Kho Ảnh Người Lạ")
        win.configure(bg=BG_MAIN)
        win.geometry("920x620")
        win.resizable(True, True)

        label_header = tk.Label(win, text="📂  Kho Ảnh Người Lạ",
                                 bg=BG_HEADER, fg="white",
                                 font=("Segoe UI", 12, "bold"))
        label_header.pack(fill="x", ipady=8)

        frame_loc = tk.Frame(win, bg=BG_CARD,
                              highlightthickness=1, highlightbackground="#DADCE0")
        frame_loc.pack(fill="x", padx=10, pady=(6, 2))

        tk.Label(frame_loc, text="🔍  Lọc theo ngày:",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=8, pady=6)

        all_files = sorted([f for f in os.listdir(folder)
                             if f.endswith(".jpg")], reverse=True)

        ngay_list = ["Tất cả"]
        seen = set()
        for f in all_files:
            try:
                t = datetime.strptime(f.replace(".jpg", ""), "%Y%m%d_%H%M%S")
                ngay = t.strftime("%d/%m/%Y")
                if ngay not in seen:
                    ngay_list.append(ngay)
                    seen.add(ngay)
            except:
                pass

        var_loc    = tk.StringVar(value="Tất cả")
        combo_ngay = ttk.Combobox(frame_loc, textvariable=var_loc,
                                   values=ngay_list, state="readonly",
                                   width=14, font=("Segoe UI", 9))
        combo_ngay.pack(side="left", padx=4)

        label_dem = tk.Label(frame_loc, text=f"Tổng: {len(all_files)} ảnh",
                              bg=BG_CARD, fg=CLR_SUB, font=("Segoe UI", 9))
        label_dem.pack(side="left", padx=12)

        anh_da_chon     = set()
        label_chon_info = tk.Label(frame_loc,
                                    text="Nhấn vào ảnh để chọn xóa",
                                    bg=BG_CARD, fg=CLR_SUB,
                                    font=("Segoe UI", 9))
        label_chon_info.pack(side="left", padx=8)

        frame_outer = tk.Frame(win, bg=BG_MAIN)
        frame_outer.pack(fill="both", expand=True, padx=10, pady=4)

        canvas    = tk.Canvas(frame_outer, bg=BG_MAIN, highlightthickness=0)
        scrollbar = tk.Scrollbar(frame_outer, orient="vertical", command=canvas.yview)
        frame_grid = tk.Frame(canvas, bg=BG_MAIN)
        frame_grid.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame_grid, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                         lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        COLS, THUMB = 4, 190
        card_widgets = {}

        def toggle_chon(filename, card):
            if filename in anh_da_chon:
                anh_da_chon.remove(filename)
                card.config(highlightbackground="#DADCE0", highlightthickness=1)
            else:
                anh_da_chon.add(filename)
                card.config(highlightbackground=CLR_RED, highlightthickness=3)
            n = len(anh_da_chon)
            label_chon_info.config(
                text=f"Đã chọn {n} ảnh" if n > 0 else "Nhấn vào ảnh để chọn xóa",
                fg=CLR_RED if n > 0 else CLR_SUB)

        def hien_anh(files_hien):
            for w in frame_grid.winfo_children():
                w.destroy()
            card_widgets.clear()
            anh_da_chon.clear()
            label_chon_info.config(text="Nhấn vào ảnh để chọn xóa", fg=CLR_SUB)
            label_dem.config(text=f"Tổng: {len(files_hien)} ảnh")
            label_header.config(text=f"📂  Kho Ảnh Người Lạ  —  {len(files_hien)} ảnh")

            for i, filename in enumerate(files_hien):
                path = os.path.join(folder, filename)
                try:
                    img   = Image.open(path).resize((THUMB, THUMB))
                    imgtk = ImageTk.PhotoImage(img)
                    row, col = i // COLS, i % COLS

                    card = tk.Frame(frame_grid, bg=BG_CARD,
                                    highlightthickness=1,
                                    highlightbackground="#DADCE0",
                                    cursor="hand2")
                    card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

                    lbl_img = tk.Label(card, image=imgtk, bg=BG_CARD, cursor="hand2")
                    lbl_img.image = imgtk
                    lbl_img.pack()

                    try:
                        t  = datetime.strptime(filename.replace(".jpg",""), "%Y%m%d_%H%M%S")
                        ts = t.strftime("%d/%m/%Y  %H:%M:%S")
                    except:
                        ts = filename

                    frame_ts = tk.Frame(card, bg="#EA4335")
                    frame_ts.pack(fill="x")
                    tk.Label(frame_ts, text=f"🕐  {ts}",
                             bg="#EA4335", fg="white",
                             font=("Segoe UI", 9, "bold"), pady=3).pack()

                    fn = filename
                    card.bind("<Button-1>",   lambda e, f=fn, c=card: toggle_chon(f, c))
                    lbl_img.bind("<Button-1>", lambda e, f=fn, c=card: toggle_chon(f, c))
                    card_widgets[filename] = card
                except Exception as e:
                    print(f"Lỗi {filename}: {e}")

        def loc_anh(*args):
            ngay = var_loc.get()
            if ngay == "Tất cả":
                hien_anh(all_files)
            else:
                hien_anh([f for f in all_files if
                           datetime.strptime(f.replace(".jpg",""), "%Y%m%d_%H%M%S"
                                             ).strftime("%d/%m/%Y") == ngay])

        combo_ngay.bind("<<ComboboxSelected>>", loc_anh)
        hien_anh(all_files)

        frame_bottom = tk.Frame(win, bg=BG_MAIN)
        frame_bottom.pack(fill="x", padx=10, pady=(4, 8))

        def xoa_da_chon():
            if not anh_da_chon:
                from tkinter import messagebox
                messagebox.showwarning("Chưa chọn",
                                        "Nhấn vào ảnh muốn xóa trước!", parent=win)
                return
            from tkinter import messagebox
            if not messagebox.askyesno("Xác nhận",
                                        f"Xóa {len(anh_da_chon)} ảnh đã chọn?",
                                        parent=win):
                return
            for fn in list(anh_da_chon):
                try:
                    os.remove(os.path.join(folder, fn))
                    all_files.remove(fn)
                except:
                    pass
            self._ghi_log(f"[OK] Đã xóa {len(anh_da_chon)} ảnh người lạ.")
            self.tong_canh_bao = max(0, self.tong_canh_bao - len(anh_da_chon))
            self.label_so_la.config(text=str(self.tong_canh_bao))
            loc_anh()

        tk.Button(frame_bottom, text="🗑  Xóa Ảnh Đã Chọn",
                  bg=CLR_RED, fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", pady=7, cursor="hand2", bd=0,
                  command=xoa_da_chon).pack(side="left", padx=4)

        tk.Button(frame_bottom, text="☑  Chọn Tất Cả",
                  bg="#37474F", fg="white", font=("Segoe UI", 9, "bold"),
                  relief="flat", pady=7, cursor="hand2", bd=0,
                  command=lambda: [
                      anh_da_chon.update(card_widgets.keys()),
                      [c.config(highlightbackground=CLR_RED, highlightthickness=3)
                       for c in card_widgets.values()],
                      label_chon_info.config(
                          text=f"Đã chọn {len(anh_da_chon)} ảnh", fg=CLR_RED)
                  ]).pack(side="left", padx=4)

        tk.Button(frame_bottom, text="✕  Bỏ Chọn Tất Cả",
                  bg="#F1F3F4", fg=CLR_GRAY, font=("Segoe UI", 9),
                  relief="flat", pady=7, cursor="hand2", bd=0,
                  command=lambda: [
                      anh_da_chon.clear(),
                      [c.config(highlightbackground="#DADCE0", highlightthickness=1)
                       for c in card_widgets.values()],
                      label_chon_info.config(
                          text="Nhấn vào ảnh để chọn xóa", fg=CLR_SUB)
                  ]).pack(side="left", padx=4)

        tk.Button(frame_bottom, text="🗑  Xóa Tất Cả",
                  bg="#B71C1C", fg="white", font=("Segoe UI", 9),
                  relief="flat", pady=7, cursor="hand2", bd=0,
                  command=lambda: self._xoa_tat_ca_nguoi_la(win)
                  ).pack(side="right", padx=4)

    def _xoa_tat_ca_nguoi_la(self, win):
        folder = "unknown_log"
        if not os.path.exists(folder):
            return
        files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        for f in files:
            os.remove(os.path.join(folder, f))
        self.tong_canh_bao = 0
        self.label_so_la.config(text="0")
        self._ghi_log(f"[OK] Đã xóa {len(files)} ảnh người lạ.")
        win.destroy()

    def xuat_log(self):
        from tkinter import filedialog
        noi_dung = self.log_text.get("1.0", "end").strip()
        if not noi_dung:
            self._ghi_log("[!] Log đang trống, không có gì để xuất.")
            return
        ten_mac_dinh = f"log_canh_bao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        duong_dan = filedialog.asksaveasfilename(
            title="Lưu file log", defaultextension=".txt",
            initialfile=ten_mac_dinh,
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")])
        if not duong_dan:
            return
        with open(duong_dan, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("  HỆ THỐNG NHẬN DIỆN NGƯỜI LẠ\n")
            f.write(f"  Xuất lúc: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(noi_dung)
        self._ghi_log(f"[OK] Đã xuất log ra: {os.path.basename(duong_dan)}", "ok")

    def xoa_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def ve_chung_toi(self):
        win = tk.Toplevel(self.root)
        win.title("Về Chúng Tôi")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)
        win.attributes("-topmost", True)

        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"420x400+{(sw-420)//2}+{(sh-400)//2}")

        tk.Label(win, text="ℹ  Về Chúng Tôi",
                 bg=BG_HEADER, fg="white",
                 font=("Segoe UI", 12, "bold")).pack(fill="x", ipady=10)

        card = self._card(win)
        card.pack(fill="both", expand=True, padx=16, pady=12)

        info = [
            ("📚  Môn học",     "Thị Giác Máy Tính"),
            ("🏫  Trường",      "Trường Đại Học Vinh"),
            ("👩‍🏫  Giáo viên",  "Nguyễn Thị Minh Tâm"),
            ("📌  Đề tài",      "Hệ thống phát hiện và\ncảnh báo người lạ vào nhà"),
            ("👥  Nhóm",        "Nhóm 6"),
        ]

        thanh_vien = [
            ("Nguyễn Văn Thân",  "MSSV: 235748020110172"),
            ("Nguyễn Trọng Mạnh", "MSSV: 235748020110259"),
            ("Hoàng Minh Thắng",  "MSSV: 235748020110039"),
        ]

        for nhan, gia_tri in info:
            f = tk.Frame(card, bg=BG_CARD)
            f.pack(fill="x", padx=12, pady=3)
            tk.Label(f, text=nhan, bg=BG_CARD, fg=CLR_SUB,
                     font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
            tk.Label(f, text=gia_tri, bg=BG_CARD, fg=CLR_TEXT,
                     font=("Segoe UI", 9, "bold"), justify="left").pack(side="left")

        tk.Frame(card, bg="#DADCE0", height=1).pack(fill="x", padx=12, pady=6)

        tk.Label(card, text="👨‍💻  Thành viên nhóm",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12)

        for ten, mssv in thanh_vien:
            f = tk.Frame(card, bg=BG_CARD)
            f.pack(fill="x", padx=12, pady=2)
            tk.Label(f, text=f"  • {ten}", bg=BG_CARD, fg=CLR_TEXT,
                     font=("Segoe UI", 9)).pack(side="left")
            tk.Label(f, text=mssv, bg=BG_CARD, fg=CLR_SUB,
                     font=("Segoe UI", 9)).pack(side="left", padx=8)

        tk.Label(card, text="🛠  Công nghệ: InsightFace + ArcFace + OpenCV",
                 bg=BG_CARD, fg=CLR_SUB,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=(8, 4))

        tk.Button(win, text="✕  Đóng",
                  bg=CLR_BLUE, fg="white",
                  font=("Segoe UI", 11, "bold"),
                  relief="flat", pady=10, cursor="hand2",
                  bd=0, command=win.destroy
                  ).pack(fill="x", padx=16, pady=(0, 16))

    def _cap_nhat_gio(self):
        self.label_gio.config(text=datetime.now().strftime("%H:%M:%S  |  %d/%m/%Y"))
        self.root.after(1000, self._cap_nhat_gio)

    def dong_app(self):
        self.dang_chay = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = App(root)
    root.protocol("WM_DELETE_WINDOW", app.dong_app)
    root.mainloop()