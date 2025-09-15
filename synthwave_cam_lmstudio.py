import io
import os
import platform
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

# --- LM Studio SDK (official) ---
# pip install lmstudio
import lmstudio as lms

# --- MLX-VLM for Qwen2.5-VL ---
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# If your LM Studio server isn't on the default host/port, set it here:
# lms.configure_default_client("localhost:1234")  # e.g., "localhost:1234"


# ---- Synthwave palette ----
SYNTH_BG = "#0d0221"
SYNTH_PANEL = "#1a0040"
SYNTH_ACCENT = "#2d0a5c"
SYNTH_NEON = "#ff2bd6"
SYNTH_CYAN = "#00ffff"
SYNTH_PURPLE = "#8a2be2"
SYNTH_TEXT = "#e8e6ff"
SYNTH_TEXT_DIM = "#b8a8d8"
SYNTH_BORDER = "#ff2bd6"
SYNTH_GLOW = "#ff2bd6"


def _pick_cv_backend():
    """Choose a sensible OpenCV backend per OS."""
    osname = platform.system()
    if osname == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if osname == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY


def _try_open_camera(index, backend):
    """Attempt to open a camera and read a frame; return an open cap or None."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def _scan_working_cameras(max_index=10):
    """Return list of indices that open and deliver at least one frame."""
    backend = _pick_cv_backend()
    working = []
    for idx in range(0, max_index + 1):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                working.append(idx)
    return working, backend


class SynthwaveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("◢ DARREN'S AI VIDEO RECORDER ◣")
        self.root.configure(bg=SYNTH_BG)
        self.root.minsize(1400, 900)

        self.backend = _pick_cv_backend()
        self.cap = None
        self.cam_index = None
        self.current_frame_bgr = None
        self.captured_image_pil = None

        # Video recording attributes
        self.is_recording = False
        self.video_frames = []
        self.video_frame_paths = []
        self.recording_start_time = None
        self.temp_dir = None

        # Analysis mode: 'image' or 'video'
        self.analysis_mode = tk.StringVar(value="image")

        # label -> model_key (for dropdown)
        self.model_options = {}

        # Initialize MLX-VLM model
        self.mlx_model = None
        self.mlx_processor = None
        self.mlx_config = None
        self.mlx_loading = False

        self._build_styles()

        # Main container with padding
        main = ttk.Frame(self.root, padding=15, style="Synth.TFrame")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Left panel: camera controls + live video + captured preview
        left = ttk.Frame(main, padding=15, style="LeftPanel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        main.columnconfigure(0, weight=2)
        main.rowconfigure(0, weight=1)

        # Camera controls section with header
        cam_header = ttk.Label(left, text="◢ CAMERA CONTROL ◣", style="SectionHeader.TLabel")
        cam_header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        cam_row = ttk.Frame(left, style="ControlPanel.TFrame")
        cam_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ttk.Label(cam_row, text="📹 CAMERA:", style="NeonLabel.TLabel").grid(row=0, column=0, padx=(8, 8), sticky="w")
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(cam_row, textvariable=self.camera_var, state="readonly", style="Neon.TCombobox")
        self.camera_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        cam_row.columnconfigure(1, weight=1)
        ttk.Button(cam_row, text="◎ RESCAN", style="NeonAlt.TButton", command=self.rescan_cameras).grid(
            row=0, column=2, padx=(0, 8)
        )
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_selected)

        # Live video preview with frame
        video_frame = ttk.Frame(left, style="VideoFrame.TFrame", padding=3)
        video_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 12))
        self.video_label = ttk.Label(video_frame, text="◢ INITIALIZING WEBCAM ◣\n\n🎥", anchor="center", style="VideoPreview.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=2)
        left.columnconfigure(0, weight=1)

        # Captured preview section
        capture_header = ttk.Label(left, text="◢ CAPTURED FRAME ◣", style="SectionHeader.TLabel")
        capture_header.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        capture_frame = ttk.Frame(left, style="CaptureFrame.TFrame", padding=3)
        capture_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.captured_label = ttk.Label(capture_frame, text="📸 NO CAPTURE YET", anchor="center", style="CapturePreview.TLabel")
        self.captured_label.grid(row=0, column=0, sticky="nsew")
        capture_frame.rowconfigure(0, weight=1)
        capture_frame.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=1)

        # Right panel: model controls and interaction
        right = ttk.Frame(main, padding=15, style="RightPanel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=3)

        # Model selection section
        model_header = ttk.Label(right, text="◢ AI MODEL SELECTION ◣", style="SectionHeader.TLabel")
        model_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        model_row = ttk.Frame(right, style="ControlPanel.TFrame")
        model_row.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        ttk.Label(model_row, text="🧠 MODEL:", style="NeonLabel.TLabel").grid(row=0, column=0, padx=(8, 8), sticky="w")
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, state="readonly", style="Neon.TCombobox")
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        model_row.columnconfigure(1, weight=1)
        ttk.Button(model_row, text="⟳ REFRESH", style="NeonAlt.TButton", command=self.refresh_models).grid(
            row=0, column=2, padx=(0, 8)
        )

        # Prompt section
        prompt_header = ttk.Label(right, text="◢ PROMPT INPUT ◣", style="SectionHeader.TLabel")
        prompt_header.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        prompt_frame = ttk.Frame(right, style="TextFrame.TFrame", padding=2)
        prompt_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 15))
        self.prompt_text = tk.Text(
            prompt_frame, height=6, wrap="word", bg=SYNTH_PANEL, fg=SYNTH_TEXT,
            insertbackground=SYNTH_CYAN, relief="flat", font=("Consolas", 11),
            selectbackground=SYNTH_PURPLE, selectforeground=SYNTH_TEXT
        )
        self.prompt_text.insert("1.0", "analyze the image.")
        self.prompt_text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        prompt_frame.rowconfigure(0, weight=1)
        prompt_frame.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        # Mode selection
        mode_header = ttk.Label(right, text="◢ ANALYSIS MODE ◣", style="SectionHeader.TLabel")
        mode_header.grid(row=4, column=0, sticky="ew", pady=(0, 8))

        mode_frame = ttk.Frame(right, style="ControlPanel.TFrame")
        mode_frame.grid(row=5, column=0, sticky="ew", pady=(0, 15))

        ttk.Radiobutton(mode_frame, text="📸 SINGLE FRAME", variable=self.analysis_mode,
                       value="image", style="NeonRadio.TRadiobutton").grid(row=0, column=0, padx=(8, 16), sticky="w")
        ttk.Radiobutton(mode_frame, text="🎬 VIDEO SEQUENCE", variable=self.analysis_mode,
                       value="video", style="NeonRadio.TRadiobutton").grid(row=0, column=1, padx=(16, 8), sticky="w")

        # Video recording controls
        video_controls = ttk.Frame(right, style="ControlPanel.TFrame")
        video_controls.grid(row=6, column=0, sticky="ew", pady=(0, 15))

        ttk.Label(video_controls, text="📽️ DURATION:", style="NeonLabel.TLabel").grid(row=0, column=0, padx=(8, 8), sticky="w")
        self.duration_var = tk.DoubleVar(value=3.0)
        self.duration_scale = ttk.Scale(video_controls, from_=1.0, to=10.0, variable=self.duration_var,
                                       orient="horizontal")
        self.duration_scale.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.duration_label = ttk.Label(video_controls, text="3.0s", style="Status.TLabel")
        self.duration_label.grid(row=0, column=2, padx=(0, 8))
        video_controls.columnconfigure(1, weight=1)

        # Update duration label
        def update_duration_label(*args):
            self.duration_label.config(text=f"{self.duration_var.get():.1f}s")
        self.duration_var.trace("w", update_duration_label)

        # Action controls
        btn_row = ttk.Frame(right, style="ActionPanel.TFrame")
        btn_row.grid(row=7, column=0, sticky="ew", pady=(0, 15))

        # Enhanced send button
        self.send_btn = ttk.Button(btn_row, text="◢ CAPTURE & ANALYZE ◣", style="MainAction.TButton",
                                   command=self.capture_and_send)
        self.send_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        # Video recording button
        self.record_btn = ttk.Button(btn_row, text="🔴 RECORD VIDEO", style="RecordButton.TButton",
                                    command=self.toggle_video_recording)
        self.record_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        # Status display
        self.status_var = tk.StringVar(value="◎ READY")
        status_frame = ttk.Frame(btn_row, style="StatusFrame.TFrame", padding=8)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(status_frame, text="STATUS:", style="StatusLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=1, sticky="w", padx=(8, 0))
        btn_row.columnconfigure(0, weight=1)

        # Response section
        response_header = ttk.Label(right, text="◢ AI RESPONSE ◣", style="SectionHeader.TLabel")
        response_header.grid(row=8, column=0, sticky="ew", pady=(0, 8))

        output_frame = ttk.Frame(right, style="TextFrame.TFrame", padding=2)
        output_frame.grid(row=9, column=0, sticky="nsew")
        self.output_text = tk.Text(
            output_frame, height=10, wrap="word", bg=SYNTH_PANEL, fg=SYNTH_TEXT,
            insertbackground=SYNTH_CYAN, relief="flat", font=("Consolas", 11),
            selectbackground=SYNTH_PURPLE, selectforeground=SYNTH_TEXT
        )
        self.output_text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        right.rowconfigure(9, weight=2)

        # Initialize cameras (scan + open first)
        self.rescan_cameras(open_first=True)

        # Start preview loop and populate models
        self._video_loop()
        self.refresh_models()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- MLX-VLM Model Initialization ----------
    def _initialize_mlx_model(self):
        """Initialize MLX-VLM model (optimized for Apple Silicon)"""
        if self.mlx_loading or (self.mlx_model is not None):
            return

        self.mlx_loading = True

        def worker():
            try:
                self.status_var.set("🧠 LOADING MLX MODEL...")

                # Use quantized model for better performance
                model_path = "mlx-community/Qwen2.5-VL-32B-Instruct-8bit"

                self.mlx_model, self.mlx_processor = load(model_path)
                self.mlx_config = self.mlx_model.config

                self.status_var.set("✓ MLX MODEL LOADED (APPLE SILICON)")
                print(f"MLX-VLM model loaded successfully: {model_path}")
            except Exception as e:
                self.status_var.set("✗ MLX MODEL FAILED")
                print(f"Failed to load MLX model: {e}")
                self.mlx_model = None
                self.mlx_processor = None
                self.mlx_config = None
            finally:
                self.mlx_loading = False

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Enhanced Styles ----------
    def _build_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Base frames with enhanced styling
        style.configure("Synth.TFrame", background=SYNTH_BG)
        style.configure("LeftPanel.TFrame", background=SYNTH_PANEL, relief="ridge", borderwidth=2)
        style.configure("RightPanel.TFrame", background=SYNTH_PANEL, relief="ridge", borderwidth=2)
        style.configure("ControlPanel.TFrame", background=SYNTH_ACCENT, relief="raised", borderwidth=1)
        style.configure("ActionPanel.TFrame", background=SYNTH_ACCENT, relief="raised", borderwidth=1)

        # Video and capture frames with neon borders
        style.configure("VideoFrame.TFrame", background=SYNTH_BORDER, relief="solid", borderwidth=2)
        style.configure("CaptureFrame.TFrame", background=SYNTH_PURPLE, relief="solid", borderwidth=2)
        style.configure("TextFrame.TFrame", background=SYNTH_BORDER, relief="solid", borderwidth=1)
        style.configure("StatusFrame.TFrame", background=SYNTH_ACCENT, relief="sunken", borderwidth=1)

        # Enhanced labels with better hierarchy
        style.configure("SectionHeader.TLabel", background=SYNTH_PANEL, foreground=SYNTH_NEON,
                       font=("Impact", 12, "bold"), anchor="center")
        style.configure("NeonLabel.TLabel", background=SYNTH_ACCENT, foreground=SYNTH_CYAN,
                       font=("Arial", 10, "bold"))
        style.configure("VideoPreview.TLabel", background=SYNTH_PANEL, foreground=SYNTH_TEXT_DIM,
                       font=("Arial", 14), anchor="center")
        style.configure("CapturePreview.TLabel", background=SYNTH_PANEL, foreground=SYNTH_TEXT_DIM,
                       font=("Arial", 12), anchor="center")
        style.configure("StatusLabel.TLabel", background=SYNTH_ACCENT, foreground=SYNTH_CYAN,
                       font=("Arial", 9, "bold"))
        style.configure("Status.TLabel", background=SYNTH_ACCENT, foreground=SYNTH_TEXT,
                       font=("Arial", 9))

        # Enhanced buttons with better visual feedback
        style.configure("MainAction.TButton", background=SYNTH_NEON, foreground="#000000",
                       font=("Impact", 12, "bold"), relief="raised", borderwidth=2)
        style.map("MainAction.TButton",
                 background=[("active", "#ff5be0"), ("pressed", "#cc1fa8"), ("disabled", "#661b52")],
                 foreground=[("disabled", "#331829")],
                 relief=[("pressed", "sunken")])

        style.configure("NeonAlt.TButton", background=SYNTH_CYAN, foreground="#000000",
                       font=("Arial", 9, "bold"), relief="raised", borderwidth=1)
        style.map("NeonAlt.TButton",
                 background=[("active", "#66ffff"), ("pressed", "#00cccc"), ("disabled", "#004d4d")],
                 foreground=[("disabled", "#002626")],
                 relief=[("pressed", "sunken")])

        # Record button styling
        style.configure("RecordButton.TButton", background="#ff0000", foreground="#ffffff",
                       font=("Impact", 10, "bold"), relief="raised", borderwidth=2)
        style.map("RecordButton.TButton",
                 background=[("active", "#ff3333"), ("pressed", "#cc0000"), ("disabled", "#660000")],
                 foreground=[("disabled", "#330000")],
                 relief=[("pressed", "sunken")])

        # Enhanced combobox styling
        style.configure("Neon.TCombobox", fieldbackground=SYNTH_PANEL, background=SYNTH_ACCENT,
                       foreground=SYNTH_TEXT, font=("Arial", 10))
        style.map("Neon.TCombobox",
                 fieldbackground=[("readonly", SYNTH_PANEL), ("focus", SYNTH_ACCENT)],
                 bordercolor=[("focus", SYNTH_CYAN)])

        # Radio button styling
        style.configure("NeonRadio.TRadiobutton", background=SYNTH_ACCENT, foreground=SYNTH_CYAN,
                       font=("Arial", 9, "bold"))
        style.map("NeonRadio.TRadiobutton",
                 background=[("active", SYNTH_PURPLE)],
                 foreground=[("active", SYNTH_TEXT)])


    # ---------- Cameras ----------
    def rescan_cameras(self, open_first=False):
        """Scan for available cameras, repopulate dropdown, optionally open the first one."""
        indices, backend = _scan_working_cameras()
        self.backend = backend
        if not indices:
            self.camera_combo["values"] = []
            self.camera_var.set("")
            self._close_cap()
            messagebox.showerror(
                "◢ WEBCAM ERROR ◣",
                "Could not access any webcam.\n\n"
                "Quick tips:\n"
                "• Close Zoom/Teams/Photo Booth/etc.\n"
                "• Allow camera access for your terminal/IDE in system privacy settings.\n"
                "• If on Linux, ensure user is in the 'video' group and /dev/video* exists."
            )
            self.status_var.set("✗ NO CAMERA FOUND")
            return

        labels = [f"Camera {i}" for i in indices]
        self.camera_combo["values"] = labels

        current_label = self.camera_var.get()
        if current_label in labels:
            pass
        else:
            self.camera_var.set(labels[0])

        if open_first:
            self._open_selected_camera()

        if self.cap:
            self.status_var.set(f"◎ READY - {self.camera_var.get()} ({self._backend_name(self.backend)})")
        else:
            self.status_var.set("⚠ SELECT CAMERA")

    def _backend_name(self, backend):
        mapping = {
            getattr(cv2, "CAP_AVFOUNDATION", -1): "AVFOUNDATION",
            getattr(cv2, "CAP_DSHOW", -1): "DSHOW",
            getattr(cv2, "CAP_V4L2", -1): "V4L2",
            getattr(cv2, "CAP_ANY", -1): "ANY",
        }
        return mapping.get(backend, str(backend))

    def _on_camera_selected(self, _event=None):
        self._open_selected_camera()

    def _open_selected_camera(self):
        label = self.camera_var.get().strip()
        if not label:
            return
        try:
            index = int(label.split()[-1])
        except Exception:
            index = 0
        self._reopen_cap(index)

    def _reopen_cap(self, index):
        """Close existing cap and open a new one at index."""
        self._close_cap()
        cap = _try_open_camera(index, self.backend)
        if cap is None:
            messagebox.showerror("Webcam Error", f"Could not open {index} with backend {self._backend_name(self.backend)}.")
            self.status_var.set("✗ CAMERA FAILED")
            return
        self.cap = cap
        self.cam_index = index
        self.status_var.set(f"◎ READY - Camera {index} ({self._backend_name(self.backend)})")

    def _close_cap(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.cam_index = None

    # ---------- Webcam loop ----------
    def _video_loop(self):
        if self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                self.current_frame_bgr = frame

                # Record frames if recording
                if self.is_recording and len(self.video_frames) < 300:  # Max 10 seconds at ~30fps
                    current_time = time.time()
                    if not hasattr(self, '_last_frame_time') or current_time - self._last_frame_time >= 0.5:  # 2 fps
                        self.video_frames.append(frame.copy())
                        self._last_frame_time = current_time

                        # Update recording status
                        elapsed = current_time - self.recording_start_time
                        remaining = max(0, self.duration_var.get() - elapsed)
                        self.status_var.set(f"🔴 RECORDING... {remaining:.1f}s")

                        # Stop recording when duration reached
                        if elapsed >= self.duration_var.get():
                            self.stop_video_recording()

                # Update video display
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = img.resize((480, int(480 * img.height / img.width))) if img.width > 480 else img
                imgtk = ImageTk.PhotoImage(image=img)

                # Add recording indicator overlay
                if self.is_recording:
                    # Add red border to indicate recording
                    display_text = ""
                else:
                    display_text = ""

                self.video_label.configure(image=imgtk, text=display_text)
                self.video_label.image = imgtk
        self.root.after(30, self._video_loop)

    # ---------- Video Recording ----------
    def toggle_video_recording(self):
        """Start or stop video recording"""
        if not self.is_recording:
            self.start_video_recording()
        else:
            self.stop_video_recording()

    def start_video_recording(self):
        """Start recording video frames"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("No Camera", "Please connect a camera first.")
            return

        # Reset recording state
        self.video_frames = []
        self.video_frame_paths = []
        self.is_recording = True
        self.recording_start_time = time.time()
        self._last_frame_time = 0

        # Create temp directory for frames
        self.temp_dir = tempfile.mkdtemp(prefix="video_frames_")

        # Update UI
        self.record_btn.configure(text="⏹️ STOP RECORDING")
        self.send_btn.configure(state="disabled")
        self.status_var.set(f"🔴 RECORDING... {self.duration_var.get():.1f}s")

    def stop_video_recording(self):
        """Stop recording and save frames"""
        if not self.is_recording:
            return

        self.is_recording = False

        # Save frames to temporary files
        self.video_frame_paths = []
        for i, frame in enumerate(self.video_frames):
            frame_path = os.path.join(self.temp_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            self.video_frame_paths.append(f"file://{frame_path}")

        # Update UI
        self.record_btn.configure(text="🔴 RECORD VIDEO")
        self.send_btn.configure(state="normal")
        self.status_var.set(f"✓ RECORDED {len(self.video_frames)} FRAMES")

        # Show preview of last frame
        if self.video_frames:
            last_frame = self.video_frames[-1]
            rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            self.captured_image_pil = Image.fromarray(rgb)
            preview = self.captured_image_pil.copy()
            preview.thumbnail((320, 320))
            imgtk = ImageTk.PhotoImage(preview)
            self.captured_label.configure(image=imgtk, text="")
            self.captured_label.image = imgtk

    # ---------- Models ----------
    def _extract_model_key_and_name_from_loaded(self, loaded_handle):
        """Return (model_key, display_name) from a loaded LLM handle via get_info()."""
        info = loaded_handle.get_info()  # includes 'vision' flag
        model_key = getattr(info, "model_key", None) or getattr(info, "modelKey", None) or getattr(info, "identifier", None)
        display = getattr(info, "display_name", None) or getattr(info, "displayName", None) or str(model_key)
        return model_key, display

    def refresh_models(self):
        def worker():
            try:
                self.status_var.set("⟳ FETCHING MODELS...")
                labels_to_keys = {}

                loaded = lms.list_loaded_models()  # LLM handles
                if loaded:
                    for h in loaded:
                        key, name = self._extract_model_key_and_name_from_loaded(h)
                        if key:
                            labels_to_keys[f"{name} [{key}]"] = key
                else:
                    downloaded = lms.list_downloaded_models()  # objects with .model_key / .display_name
                    for d in downloaded:
                        key = getattr(d, "model_key", None)
                        name = getattr(d, "display_name", None) or key
                        if key:
                            labels_to_keys[f"{name} [{key}]"] = key

                if not labels_to_keys:
                    raise RuntimeError("No models found. Download one in LM Studio and/or load it into memory.")

                self.root.after(0, lambda: self._update_models_ui(labels_to_keys))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Model Error", f"Could not list models:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("✗ MODEL FETCH FAILED"))
        threading.Thread(target=worker, daemon=True).start()

    def _update_models_ui(self, labels_to_keys: dict):
        self.model_options = labels_to_keys
        labels = list(labels_to_keys.keys())
        self.model_combo["values"] = labels
        current_label = self.model_var.get()
        if current_label in labels:
            self.model_combo.set(current_label)
        else:
            self.model_combo.set(labels[0])
        self.status_var.set("✓ MODELS UPDATED")

    # ---------- Capture & Send ----------
    def capture_and_send(self):
        """Capture current frame or use recorded video, then analyze."""
        # Ensure prompt isn't empty
        if not self.prompt_text.get("1.0", "end").strip():
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", "analyze the content.")

        mode = self.analysis_mode.get()

        if mode == "image":
            # Single frame analysis (existing logic)
            if self.current_frame_bgr is not None:
                rgb = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2RGB)
                self.captured_image_pil = Image.fromarray(rgb)
                # Update preview
                preview = self.captured_image_pil.copy()
                preview.thumbnail((320, 320))
                imgtk = ImageTk.PhotoImage(preview)
                self.captured_label.configure(image=imgtk, text="")
                self.captured_label.image = imgtk
                self.status_var.set("📸 FRAME CAPTURED")
            else:
                self.captured_image_pil = None
                self.captured_label.configure(image="", text="📸 NO CAPTURE YET")
                self.captured_label.image = None
                messagebox.showwarning("No Frame", "No video frame available; sending text only.")

            # Send via LM Studio
            self._send_to_llm()

        elif mode == "video":
            # Video sequence analysis
            if not self.video_frames:
                messagebox.showwarning("No Video", "Please record a video sequence first.")
                return

            # Send via MLX-VLM
            self._send_to_mlx()

    def _send_to_llm(self):
        label = self.model_var.get().strip()
        model_key = self.model_options.get(label) if hasattr(self, "model_options") else None
        if not model_key:
            messagebox.showwarning("No Model", "Please select a model.")
            return

        user_text = self.prompt_text.get("1.0", "end").strip()
        if not user_text:
            user_text = "analyze the image."  # safety net

        wants_image = self.captured_image_pil is not None

        self.send_btn.config(state="disabled")
        self.status_var.set("🚀 ANALYZING...")

        def worker():
            tmp_path = None
            try:
                model = lms.llm(model_key)

                info = model.get_info()
                supports_vision = bool(getattr(info, "vision", False))

                chat = lms.Chat()
                images = None

                if wants_image:
                    if supports_vision:
                        # Temp PNG so LM Studio logs a proper filename/type
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp_path = tmp.name
                            self.captured_image_pil.save(tmp, format="PNG")
                            tmp.flush()
                        image_handle = lms.prepare_image(tmp_path)
                        images = [image_handle]
                    else:
                        self.root.after(0, lambda: messagebox.showwarning(
                            "Model Lacks Vision",
                            "The selected model does not support image input. "
                            "Your text prompt will be sent without the image.\n\n"
                            "Tip: try a VLM like 'qwen2-vl-2b-instruct'."
                        ))

                chat.add_user_message(user_text, images=images)
                result = model.respond(chat, config={"temperature": 0.2})
                text = getattr(result, "content", None) or str(result)

                self.root.after(0, lambda: self._set_output(text))
                self.root.after(0, lambda: self.status_var.set("✓ ANALYSIS COMPLETE"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Request Error", f"LLM request failed:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("✗ REQUEST FAILED"))
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                self.root.after(0, lambda: self.send_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _send_to_mlx(self):
        """Send video frames to MLX-VLM for analysis"""
        if not self.mlx_model or not self.mlx_processor:
            if not self.mlx_loading:
                self._initialize_mlx_model()
                messagebox.showinfo("Loading Model", "MLX-VLM model is loading. Please wait and try again.")
                return
            else:
                messagebox.showinfo("Loading Model", "MLX-VLM model is still loading. Please wait.")
                return

        user_text = self.prompt_text.get("1.0", "end").strip()
        if not user_text:
            user_text = "Describe what happens in this video sequence."

        self.send_btn.config(state="disabled")
        self.status_var.set("🧠 ANALYZING VIDEO...")

        def worker():
            try:
                # Convert file:// paths to regular paths for MLX-VLM
                image_paths = []
                for frame_path in self.video_frame_paths:
                    if frame_path.startswith("file://"):
                        image_paths.append(frame_path[7:])  # Remove 'file://' prefix
                    else:
                        image_paths.append(frame_path)

                # Apply chat template for multiple images
                formatted_prompt = apply_chat_template(
                    self.mlx_processor,
                    self.mlx_config,
                    user_text,
                    num_images=len(image_paths)
                )

                # Generate response using MLX-VLM
                output = generate(
                    self.mlx_model,
                    self.mlx_processor,
                    formatted_prompt,
                    image_paths,
                    verbose=False
                )

                self.root.after(0, lambda: self._set_output(output))
                self.root.after(0, lambda: self.status_var.set("✓ VIDEO ANALYSIS COMPLETE"))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Video Analysis Error", f"MLX analysis failed:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("✗ VIDEO ANALYSIS FAILED"))
                print(f"MLX analysis error: {e}")
            finally:
                self.root.after(0, lambda: self.send_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _set_output(self, text):
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)

    def on_close(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

        # Clean up temporary files
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass

        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1400x900")
    SynthwaveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

