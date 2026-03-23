import cv2
import tkinter as tk
from tkinter import scrolledtext, font
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk
import pickle
import os
from insightface.app import FaceAnalysis

# ── Cấu hình ─────────────────────────────────────────
ENCODING_FILE = "encodings.pkl"
NGUONG_COSINE = 0.5
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

    # Vẽ thông tin lên ảnh trước khi lưu
    anh_luu = frame.copy()

    # Vẽ khung đỏ cho người lạ, xanh cho người quen
    for (box, ten, do_cx, mau) in ket_qua:
        cv2.rectangle(anh_luu,
                      (box[0], box[1]), (box[2], box[3]), mau, 2)
        cv2.rectangle(anh_luu,
                      (box[0], box[3] - 32), (box[2], box[3]),
                      mau, cv2.FILLED)
        nhan = ten if ten == "NGUOI LA" else f"{ten} ({do_cx:.0f}%)"
        cv2.putText(anh_luu, nhan,
                    (box[0] + 4, box[3] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

    # Vẽ timestamp góc trên trái
    gio = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
    cv2.rectangle(anh_luu, (0, 0), (280, 28), (0, 0, 0), cv2.FILLED)
    cv2.putText(anh_luu, gio,
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 1)

    cv2.imwrite(f"unknown_log/{ten_file}", anh_luu)
    return ten_file

class App:
    def __init__(self, root):
        self.root  = root
        self.root.title("He thong nhan dien nguoi la")
        self.root.configure(bg=BG_MAIN)
        self.root.resizable(False, False)

        # Căn giữa màn hình với kích thước cố định
        w, h = 980, 620
        sw   = self.root.winfo_screenwidth()
        sh   = self.root.winfo_screenheight()
        x    = (sw - w) // 2
        y    = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

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

        self._build_ui()
        self._load_model_thread()

    def _card(self, parent, **kwargs):
        return tk.Frame(parent, bg=BG_CARD,
                        relief="flat",
                        highlightthickness=1,
                        highlightbackground="#DADCE0",
                        **kwargs)

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=BG_HEADER, height=52)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header,
                 text="  🏠  He Thong Nhan Dien Nguoi La",
                 bg=BG_HEADER, fg="white",
                 font=("Arial", 14, "bold")).pack(side="left", pady=10)

        self.label_gio = tk.Label(header, text="",
                                   bg=BG_HEADER, fg="#E8F0FE",
                                   font=("Arial", 10))
        self.label_gio.pack(side="right", padx=16)
        self._cap_nhat_gio()

        # ── Body ──
        body = tk.Frame(self.root, bg=BG_MAIN)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # ═══ CỘT TRÁI: Camera ═══
        col_trai = tk.Frame(body, bg=BG_MAIN)
        col_trai.pack(side="left", fill="both", expand=True)

        # Card camera
        card_cam = self._card(col_trai)
        card_cam.pack(fill="both", expand=True, pady=(0, 8))

        tk.Label(card_cam,
                 text="📷  Camera Giam Sat",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Arial", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        tk.Frame(card_cam, bg="#DADCE0", height=1).pack(fill="x", padx=12)

        # Frame cố định pixel cho camera
        frame_cam_inner = tk.Frame(card_cam, bg="#000000",
                                    width=640, height=480)
        frame_cam_inner.pack(padx=12, pady=8)
        frame_cam_inner.pack_propagate(False)  # giữ nguyên kích thước

        self.label_camera = tk.Label(frame_cam_inner, bg="#000000")
        self.label_camera.pack(fill="both", expand=True)
        self.label_camera.pack_forget()  # reset trước khi pack lại bên dưới
        self.label_camera.pack(fill="both", expand=True)

        # Card thống kê dưới camera
        card_stat = self._card(col_trai)
        card_stat.pack(fill="x")

        frame_stat = tk.Frame(card_stat, bg=BG_CARD)
        frame_stat.pack(fill="x", padx=12, pady=8)

        stats = [
            ("Khuon mat",   "0", CLR_BLUE,   "label_so_mat"),
            ("Nguoi quen",  "0", CLR_GREEN,  "label_so_quen"),
            ("Nguoi la",    "0", CLR_RED,    "label_so_la"),
            ("DB (anh)",    "...", CLR_ORANGE, "label_db"),
        ]
        for i, (nhan, val, mau, attr) in enumerate(stats):
            f = tk.Frame(frame_stat, bg=BG_CARD)
            f.pack(side="left", expand=True)
            lbl_val = tk.Label(f, text=val,
                                bg=BG_CARD, fg=mau,
                                font=("Arial", 20, "bold"))
            lbl_val.pack()
            tk.Label(f, text=nhan,
                     bg=BG_CARD, fg=CLR_SUB,
                     font=("Arial", 8)).pack()
            setattr(self, attr, lbl_val)
            if i < len(stats) - 1:
                tk.Frame(frame_stat, bg="#DADCE0",
                         width=1).pack(side="left", fill="y", padx=8)

        # ═══ CỘT PHẢI: Điều khiển ═══
        col_phai = tk.Frame(body, bg=BG_MAIN, width=260)
        col_phai.pack(side="left", fill="y", padx=(10, 0))
        col_phai.pack_propagate(False)

        # Card trạng thái
        card_tt = self._card(col_phai)
        card_tt.pack(fill="x", pady=(0, 8))

        tk.Label(card_tt, text="⚡  Trang Thai He Thong",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Arial", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Frame(card_tt, bg="#DADCE0", height=1).pack(fill="x", padx=12)

        self.label_trang_thai = tk.Label(
            card_tt, text="Dang khoi tao...",
            bg=CLR_LIGHT, fg=CLR_BLUE,
            font=("Arial", 9),
            relief="flat", padx=8, pady=6,
            wraplength=220, justify="left"
        )
        self.label_trang_thai.pack(fill="x", padx=12, pady=8)

        # Card điều khiển
        card_dk = self._card(col_phai)
        card_dk.pack(fill="x", pady=(0, 8))

        tk.Label(card_dk, text="🎮  Dieu Khien",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Arial", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Frame(card_dk, bg="#DADCE0", height=1).pack(fill="x", padx=12)

        frame_btn = tk.Frame(card_dk, bg=BG_CARD)
        frame_btn.pack(fill="x", padx=12, pady=8)

        self.btn_bat_dau = tk.Button(
            frame_btn, text="▶  Bat Dau",
            bg=CLR_GREEN, fg="white",
            font=("Arial", 10, "bold"),
            relief="flat", pady=7, cursor="hand2",
            state="disabled", bd=0,
            activebackground="#2D9142",
            command=self.bat_dau
        )
        self.btn_bat_dau.pack(fill="x", pady=2)

        self.btn_dung = tk.Button(
            frame_btn, text="■  Dung Lai",
            bg=CLR_RED, fg="white",
            font=("Arial", 10, "bold"),
            relief="flat", pady=7, cursor="hand2",
            state="disabled", bd=0,
            activebackground="#C5221F",
            command=self.dung_lai
        )
        self.btn_dung.pack(fill="x", pady=2)

        tk.Button(
            frame_btn, text="📷  Chup Anh Nguoi Quen",
            bg=CLR_BLUE, fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", pady=6, cursor="hand2",
            bd=0, activebackground="#1557B0",
            command=self.chup_anh_nguoi_quen
        ).pack(fill="x", pady=2)

        tk.Button(
            frame_btn, text="🖼  Them Anh Tu May",
            bg="#7B1FA2", fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", pady=6, cursor="hand2",
            bd=0, activebackground="#6A1B9A",
            command=self.them_anh_tu_may
        ).pack(fill="x", pady=2)

        tk.Button(
            frame_btn, text="👥  Quan Ly Nguoi Quen",
            bg="#37474F", fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", pady=6, cursor="hand2",
            bd=0, activebackground="#263238",
            command=self.quan_ly_nguoi_quen
        ).pack(fill="x", pady=2)

        tk.Button(
            frame_btn, text="📂  Xem Anh Nguoi La",
            bg="#FF6D00", fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", pady=6, cursor="hand2",
            bd=0, activebackground="#E65100",
            command=self.xem_anh_nguoi_la
        ).pack(fill="x", pady=2)

        tk.Button(
            frame_btn, text="🗑  Xoa Log",
            bg="#F1F3F4", fg=CLR_GRAY,
            font=("Arial", 9),
            relief="flat", pady=6, cursor="hand2",
            bd=0,
            command=self.xoa_log
        ).pack(fill="x", pady=2)

        # Card log
        card_log = self._card(col_phai)
        card_log.pack(fill="both", expand=True)

        tk.Label(card_log, text="🔔  Log Canh Bao",
                 bg=BG_CARD, fg=CLR_TEXT,
                 font=("Arial", 10, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Frame(card_log, bg="#DADCE0", height=1).pack(fill="x", padx=12)

        self.log_text = scrolledtext.ScrolledText(
            card_log, width=30, height=12,
            bg="#FAFAFA", fg=CLR_TEXT,
            font=("Consolas", 8),
            relief="flat", state="disabled",
            padx=6, pady=4
        )
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _load_model_thread(self):
        self._ghi_log("Dang khoi tao InsightFace...")

        def _load():
            try:
                self.app_insight = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider",
                               "CPUExecutionProvider"]
                )
                self.app_insight.prepare(ctx_id=0, det_size=(640, 640))
                self.known_embeddings, self.known_names = load_encodings()
                so_nguoi = len(set(self.known_names))
                so_anh   = len(self.known_embeddings)

                def _update():
                    self.label_db.config(text=str(so_anh))
                    self.label_so_quen.config(text=str(so_nguoi))
                    self.label_trang_thai.config(
                        text=f"San sang! DB: {so_anh} anh / {so_nguoi} nguoi",
                        bg="#E6F4EA", fg=CLR_GREEN)
                    self.btn_bat_dau.config(state="normal")
                    self._ghi_log(
                        f"[OK] Model san sang")
                    self._ghi_log(
                        f"[OK] {so_anh} anh / {so_nguoi} nguoi quen")

                self.root.after(0, _update)
            except Exception as e:
                self.root.after(0, lambda: self._ghi_log(f"[LOI] {e}"))

        threading.Thread(target=_load, daemon=True).start()

    def bat_dau(self):
        if not self.known_embeddings:
            self._ghi_log("[!] Chua co encoding!")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._ghi_log("[!] Khong mo duoc camera!")
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
        self.label_trang_thai.config(
            text="Dang theo doi...",
            bg=CLR_LIGHT, fg=CLR_BLUE)
        self._ghi_log("[>>] Bat dau theo doi!")
        self._cap_nhat_camera()

    def dung_lai(self):
        self.dang_chay = False
        if self.cap:
            self.cap.release()
        self.label_camera.config(image="", bg="#000000")
        self.btn_bat_dau.config(state="normal")
        self.btn_dung.config(state="disabled")
        self.label_trang_thai.config(
            text="Da dung.", bg="#FCE8E6", fg=CLR_RED)
        self.label_so_mat.config(text="0")
        self._ghi_log("[■] Da dung theo doi.")

    def chup_anh_nguoi_quen(self):
        """Mở cửa sổ chụp ảnh người quen"""
        # Dừng camera chính nếu đang chạy
        tam_dung = self.dang_chay
        if self.dang_chay:
            self.dang_chay = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.label_camera.config(image="", bg="#000000")
            self._ghi_log("[~] Tam dung camera chinh.")

        win = tk.Toplevel(self.root)
        win.title("Chup Anh Nguoi Quen")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)

        # Header
        tk.Label(win, text="📷  Chup Anh Nguoi Quen",
                 bg=BG_HEADER, fg="white",
                 font=("Arial", 12, "bold")).pack(fill="x", ipady=8)

        # Nhập tên
        frame_ten = tk.Frame(win, bg=BG_MAIN)
        frame_ten.pack(fill="x", padx=12, pady=6)
        tk.Label(frame_ten, text="Ten nguoi:",
                 bg=BG_MAIN, fg=CLR_TEXT,
                 font=("Arial", 10)).pack(side="left")
        entry_ten = tk.Entry(frame_ten, font=("Arial", 10), width=20)
        entry_ten.pack(side="left", padx=8)
        entry_ten.focus()

        # Camera — dùng frame cố định pixel
        frame_cam = tk.Frame(win, bg="#000000", width=640, height=420)
        frame_cam.pack(padx=12, pady=4)
        frame_cam.pack_propagate(False)
        label_preview = tk.Label(frame_cam, bg="#000000")
        label_preview.pack(fill="both", expand=True)

        # Info
        label_info = tk.Label(win,
                               text="Nhap ten → Bat Dau → Nhan S de chup",
                               bg=BG_MAIN, fg=CLR_BLUE,
                               font=("Arial", 9, "bold"))
        label_info.pack(pady=2)

        state = {
            "chup": False, "cap": None,
            "dem": 0, "so_cu": 0,
            "ten": "", "frame": None
        }

        def bat_dau():
            ten = entry_ten.get().strip().replace(" ", "_")
            if not ten:
                label_info.config(text="Hay nhap ten!", fg=CLR_RED)
                return
            state["ten"]  = ten
            state["dem"]  = 0
            state["chup"] = True
            os.makedirs(f"known_faces/{ten}", exist_ok=True)
            state["so_cu"] = len([
                f for f in os.listdir(f"known_faces/{ten}")
                if f.endswith(".jpg")
            ])
            state["cap"] = cv2.VideoCapture(0)
            state["cap"].set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            state["cap"].set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
            btn_bat_dau.config(state="disabled")
            btn_chup.config(state="normal")
            label_info.config(
                text=f"Nhan S de chup | 0 anh", fg=CLR_GREEN)
            _loop()

        def _loop():
            if not state["chup"] or not state["cap"]:
                return
            ret, frame = state["cap"].read()
            if ret:
                state["frame"] = frame
                img   = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ).resize((640, 420))
                imgtk = ImageTk.PhotoImage(img)
                label_preview.imgtk = imgtk
                label_preview.config(image=imgtk)
            win.after(30, _loop)

        def chup():
            if not state["chup"] or state["frame"] is None:
                return
            so  = state["so_cu"] + state["dem"] + 1
            p   = f"known_faces/{state['ten']}/{state['ten']}_{so}.jpg"
            Image.fromarray(
                cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB)
            ).save(p, format="JPEG", quality=95)
            state["dem"] += 1
            label_info.config(
                text=f"Da chup {state['dem']} anh | Nhan S tiep tuc",
                fg=CLR_GREEN)

        def luu_dong():
            state["chup"] = False
            if state["cap"]:
                state["cap"].release()
            if state["dem"] > 0:
                self._ghi_log(
                    f"[OK] Da chup {state['dem']} anh cho '{state['ten']}'")
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

        btn_bat_dau = tk.Button(
            frame_btn, text="▶  Bat Dau",
            bg=CLR_GREEN, fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", pady=5, cursor="hand2",
            bd=0, command=bat_dau)
        btn_bat_dau.pack(side="left", padx=4)

        btn_chup = tk.Button(
            frame_btn, text="📸  Chup! (S)",
            bg=CLR_ORANGE, fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", pady=5, cursor="hand2",
            state="disabled", bd=0, command=chup)
        btn_chup.pack(side="left", padx=4)

        tk.Button(
            frame_btn, text="💾  Luu & Dong",
            bg=CLR_BLUE, fg="white",
            font=("Arial", 9, "bold"),
            relief="flat", pady=5, cursor="hand2",
            bd=0, command=luu_dong).pack(side="left", padx=4)

        tk.Button(
            frame_btn, text="✕  Huy",
            bg="#F1F3F4", fg=CLR_GRAY,
            font=("Arial", 9),
            relief="flat", pady=5, cursor="hand2",
            bd=0, command=huy).pack(side="left", padx=4)

        win.protocol("WM_DELETE_WINDOW", huy)

    def them_anh_tu_may(self):
        """Chọn ảnh từ máy tính, tự động xử lý vào known_faces"""
        from tkinter import filedialog, simpledialog

        # Hỏi tên người
        ten = simpledialog.askstring(
            "Ten nguoi",
            "Nhap ten nguoi (khong dau, khong cach):",
            parent=self.root
        )
        if not ten or not ten.strip():
            return
        ten = ten.strip().replace(" ", "_")

        # Chọn nhiều ảnh
        files = filedialog.askopenfilenames(
            title=f"Chon anh cho '{ten}'",
            filetypes=[("Anh", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not files:
            return

        folder = f"known_faces/{ten}"
        os.makedirs(folder, exist_ok=True)

        # Đếm ảnh cũ
        so_cu = len([f for f in os.listdir(folder)
                     if f.endswith((".jpg", ".png"))])

        # Copy ảnh vào folder
        dem = 0
        for i, src in enumerate(files):
            dst = f"{folder}/{ten}_{so_cu + i + 1}.jpg"
            try:
                img = Image.open(src).convert("RGB")
                img.save(dst, format="JPEG", quality=95)
                dem += 1
            except Exception as e:
                self._ghi_log(f"[!] Loi copy {src}: {e}")

        self._ghi_log(f"[OK] Da them {dem} anh cho '{ten}'")

        # Hiện popup xác nhận
        from tkinter import messagebox
        messagebox.showinfo(
            "Them anh thanh cong!",
            f"Da them {dem} anh cho '{ten}'\n"
            f"Dang cap nhat encoding...",
            parent=self.root
        )

        # Cập nhật encodings ngay
        self._cap_nhat_encoding_silent()

    def _cap_nhat_encoding_silent(self):
        """Chạy main.py --update trong background thread"""
        self._ghi_log("[~] Dang cap nhat encoding...")

        def _run():
            try:
                import subprocess, sys
                python = sys.executable
                result = subprocess.run(
                    [python, "main.py", "--update"],
                    capture_output=True, text=True, cwd=os.getcwd()
                )
                # Reload encodings vào bộ nhớ
                self.known_embeddings, self.known_names = load_encodings()
                so_nguoi = len(set(self.known_names))
                so_anh   = len(self.known_embeddings)

                def _ui():
                    self.label_db.config(text=str(so_anh))
                    self.label_so_quen.config(text=str(so_nguoi))
                    self._ghi_log(
                        f"[OK] Cap nhat xong: "
                        f"{so_anh} anh / {so_nguoi} nguoi")
                self.root.after(0, _ui)
            except Exception as e:
                self.root.after(0,
                    lambda: self._ghi_log(f"[LOI] {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def cap_nhat_encoding(self):
        self._cap_nhat_encoding_silent()

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
                ten, do_cx = nhan_dien(
                    face.normed_embedding,
                    self.known_embeddings,
                    self.known_names
                )
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
                    self.bo_dem_on_dinh[key] = \
                        self.bo_dem_on_dinh.get(key, 0) + 1
                self.ket_qua_hien_tai = [
                    item for item in ket_qua_moi
                    if self.bo_dem_on_dinh.get(item[1], 0) >= DEM_ON_DINH
                ]

            now = datetime.now().timestamp()
            if co_nguoi_la and (now - self.thoi_gian_canh_bao) > CANH_BAO_MOI:
                ten_file = luu_nguoi_la(frame, ket_qua_moi)
                self.tong_canh_bao += 1
                self.thoi_gian_canh_bao = now
                gio = datetime.now().strftime("%H:%M:%S")
                self._ghi_log(f"[!!!] {gio} — NGUOI LA phat hien!")
                self.label_so_la.config(text=str(self.tong_canh_bao))

            self.label_so_mat.config(
                text=str(len(self.ket_qua_hien_tai)))

        for (box, ten, do_cx, mau) in self.ket_qua_hien_tai:
            cv2.rectangle(frame,
                          (box[0], box[1]), (box[2], box[3]), mau, 2)
            cv2.rectangle(frame,
                          (box[0], box[3] - 32), (box[2], box[3]),
                          mau, cv2.FILLED)
            nhan = ten if ten == "NGUOI LA" else f"{ten} ({do_cx:.0f}%)"
            cv2.putText(frame, nhan,
                        (box[0] + 4, box[3] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

        # Lấy đúng kích thước label camera hiện tại
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

    def _ghi_log(self, msg):
        self.log_text.config(state="normal")
        gio = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"{gio}  {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def quan_ly_nguoi_quen(self):
        """Cửa sổ quản lý — xem và xóa người quen"""
        from tkinter import messagebox

        folder = "known_faces"
        if not os.path.exists(folder):
            self._ghi_log("[!] Chua co nguoi quen nao.")
            return

        nguoi_list = [
            d for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ]

        if not nguoi_list:
            self._ghi_log("[!] Danh sach nguoi quen trong.")
            return

        win = tk.Toplevel(self.root)
        win.title("Quan Ly Nguoi Quen")
        win.configure(bg=BG_MAIN)
        win.geometry("500x500")
        win.resizable(False, False)

        # Header
        tk.Label(win,
                 text=f"👥  Quan Ly Nguoi Quen  —  {len(nguoi_list)} nguoi",
                 bg=BG_HEADER, fg="white",
                 font=("Arial", 12, "bold")).pack(fill="x", ipady=10)

        tk.Label(win,
                 text="Chon nguoi muon xoa roi nhan nut Xoa:",
                 bg=BG_MAIN, fg=CLR_SUB,
                 font=("Arial", 9)).pack(anchor="w", padx=12, pady=(8, 4))

        # Frame danh sách
        frame_list = tk.Frame(win, bg=BG_MAIN)
        frame_list.pack(fill="both", expand=True, padx=12, pady=4)

        scrollbar = tk.Scrollbar(frame_list)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(
            frame_list,
            font=("Arial", 11),
            selectmode="single",
            bg=BG_CARD, fg=CLR_TEXT,
            selectbackground=CLR_BLUE,
            selectforeground="white",
            relief="flat",
            yscrollcommand=scrollbar.set,
            highlightthickness=1,
            highlightbackground="#DADCE0"
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        # Điền danh sách người quen
        def refresh_list():
            listbox.delete(0, "end")
            nguoi_list2 = [
                d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))
            ]
            for ten in sorted(nguoi_list2):
                so_anh = len([
                    f for f in os.listdir(os.path.join(folder, ten))
                    if f.endswith((".jpg", ".png"))
                ])
                listbox.insert("end", f"  👤  {ten}  ({so_anh} anh)")
            return nguoi_list2

        ds = refresh_list()

        # Thông tin chọn
        label_chon = tk.Label(win,
                               text="Chua chon ai",
                               bg=BG_MAIN, fg=CLR_SUB,
                               font=("Arial", 9))
        label_chon.pack(pady=4)

        def on_select(evt):
            sel = listbox.curselection()
            if sel:
                ten = ds[sel[0]]
                label_chon.config(
                    text=f"Da chon: {ten}",
                    fg=CLR_RED)

        listbox.bind("<<ListboxSelect>>", on_select)

        def xoa_nguoi():
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning(
                    "Chua chon", "Hay chon nguoi muon xoa!",
                    parent=win)
                return

            ten = ds[sel[0]]
            xac_nhan = messagebox.askyesno(
                "Xac nhan xoa",
                f"Ban co chac muon xoa '{ten}'?\n"
                f"Tat ca anh cua nguoi nay se bi xoa!",
                parent=win
            )
            if not xac_nhan:
                return

            # Xóa thư mục ảnh
            import shutil
            shutil.rmtree(os.path.join(folder, ten))
            self._ghi_log(f"[OK] Da xoa anh cua '{ten}'.")

            # Xóa trực tiếp khỏi bộ nhớ — không xóa người khác!
            if os.path.exists(ENCODING_FILE):
                with open(ENCODING_FILE, "rb") as f:
                    data = pickle.load(f)

                # Lọc bỏ embeddings của người bị xóa
                moi_emb  = []
                moi_name = []
                moi_daxl = set()
                for emb, name, path_xl in zip(
                    data["embeddings"],
                    data["names"],
                    data.get("da_xu_ly", [])
                ):
                    if name != ten:
                        moi_emb.append(emb)
                        moi_name.append(name)
                        moi_daxl.add(path_xl)

                # Lưu pkl mới không có người bị xóa
                with open(ENCODING_FILE, "wb") as f:
                    pickle.dump({
                        "embeddings": moi_emb,
                        "names":      moi_name,
                        "da_xu_ly":   moi_daxl
                    }, f)

                # Reload vào bộ nhớ app NGAY LẬP TỨC
                self.known_embeddings = list(moi_emb)
                self.known_names      = list(moi_name)

                so_nguoi = len(set(self.known_names))
                so_anh   = len(self.known_embeddings)

                # Cập nhật UI
                self.label_db.config(text=str(so_anh))
                self.label_so_quen.config(text=str(so_nguoi))
                self.label_trang_thai.config(
                    text=f"Da xoa '{ten}'\nCon lai: {so_anh} anh / {so_nguoi} nguoi",
                    bg="#E6F4EA", fg=CLR_GREEN)
                self._ghi_log(
                    f"[OK] Con lai: {so_anh} anh / {so_nguoi} nguoi")

            # Refresh list
            ds.clear()
            ds.extend([
                d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))
            ])
            refresh_list()
            label_chon.config(text="Da xoa thanh cong!", fg=CLR_GREEN)

        tk.Button(win,
                  text="🗑  Xoa Nguoi Nay",
                  bg=CLR_RED, fg="white",
                  font=("Arial", 10, "bold"),
                  relief="flat", pady=8, cursor="hand2",
                  bd=0, command=xoa_nguoi
                  ).pack(fill="x", padx=12, pady=(4, 2))

        tk.Button(win,
                  text="✕  Dong",
                  bg="#F1F3F4", fg=CLR_GRAY,
                  font=("Arial", 9),
                  relief="flat", pady=6, cursor="hand2",
                  bd=0, command=win.destroy
                  ).pack(fill="x", padx=12, pady=(0, 12))

    def xem_anh_nguoi_la(self):
        folder = "unknown_log"
        if not os.path.exists(folder):
            self._ghi_log("[!] Chua co anh nguoi la nao.")
            return

        files = sorted([
            f for f in os.listdir(folder)
            if f.endswith(".jpg")
        ], reverse=True)  # mới nhất lên đầu

        if not files:
            self._ghi_log("[!] Thu muc unknown_log trong.")
            return

        # Mở cửa sổ gallery
        win = tk.Toplevel(self.root)
        win.title(f"Kho Anh Nguoi La ({len(files)} anh)")
        win.configure(bg=BG_MAIN)
        win.geometry("860x580")
        win.resizable(True, True)

        # Header
        tk.Label(win,
                 text=f"📂  Kho Anh Nguoi La  —  {len(files)} anh",
                 bg=BG_HEADER, fg="white",
                 font=("Arial", 12, "bold")
                 ).pack(fill="x", ipady=10)

        # Frame cuộn
        frame_outer = tk.Frame(win, bg=BG_MAIN)
        frame_outer.pack(fill="both", expand=True, padx=10, pady=10)

        canvas   = tk.Canvas(frame_outer, bg=BG_MAIN,
                              highlightthickness=0)
        scrollbar = tk.Scrollbar(frame_outer, orient="vertical",
                                  command=canvas.yview)
        frame_grid = tk.Frame(canvas, bg=BG_MAIN)

        frame_grid.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=frame_grid, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Hiển thị ảnh dạng grid 4 cột
        COLS     = 4
        THUMB_W  = 180
        THUMB_H  = 135

        for i, filename in enumerate(files):
            path = os.path.join(folder, filename)
            try:
                img   = Image.open(path).resize((THUMB_W, THUMB_H))
                imgtk = ImageTk.PhotoImage(img)

                row = i // COLS
                col = i % COLS

                # Card từng ảnh
                card = tk.Frame(frame_grid, bg=BG_CARD,
                                highlightthickness=1,
                                highlightbackground="#DADCE0")
                card.grid(row=row, column=col,
                          padx=6, pady=6, sticky="nsew")

                lbl_img = tk.Label(card, image=imgtk, bg=BG_CARD)
                lbl_img.image = imgtk  # giữ reference
                lbl_img.pack()

                # Tên file = thời gian phát hiện
                try:
                    t = datetime.strptime(
                        filename.replace(".jpg", ""), "%Y%m%d_%H%M%S")
                    ten_hien = t.strftime("%d/%m/%Y  %H:%M:%S")
                except:
                    ten_hien = filename

                # Nền đỏ nổi bật cho timestamp
                frame_time = tk.Frame(card, bg="#EA4335")
                frame_time.pack(fill="x")
                tk.Label(frame_time,
                         text=f"🕐  {ten_hien}",
                         bg="#EA4335", fg="white",
                         font=("Arial", 9, "bold"),
                         pady=4).pack()

            except Exception as e:
                print(f"Loi load anh {filename}: {e}")

        # Nút xóa toàn bộ
        tk.Button(win,
                  text="🗑  Xoa Tat Ca Anh Nguoi La",
                  bg=CLR_RED, fg="white",
                  font=("Arial", 10, "bold"),
                  relief="flat", pady=8, cursor="hand2",
                  bd=0,
                  command=lambda: self._xoa_tat_ca_nguoi_la(win)
                  ).pack(fill="x", padx=10, pady=(0, 10))

    def _xoa_tat_ca_nguoi_la(self, win):
        folder = "unknown_log"
        if not os.path.exists(folder):
            return
        files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        for f in files:
            os.remove(os.path.join(folder, f))
        self.tong_canh_bao = 0
        self.label_so_la.config(text="0")
        self._ghi_log(f"[OK] Da xoa {len(files)} anh nguoi la.")
        win.destroy()

    def xoa_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _cap_nhat_gio(self):
        gio = datetime.now().strftime("%H:%M:%S  |  %d/%m/%Y")
        self.label_gio.config(text=gio)
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