import base64
import io
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

# OpenAI SDK, pointed at LM Studio's local server
try:
    from openai import OpenAI
except Exception as e:
    raise SystemExit(
        "The 'openai' package is required. Did you install requirements.txt?\n"
        "pip install -r requirements.txt"
    )

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"  # default LM Studio REST API
LMSTUDIO_API_KEY = "lm-studio"                  # placeholder; LM Studio ignores it

SYNTH_BG = "#1b0049"      # deep purple
SYNTH_PANEL = "#25006e"   # darker neon purple
SYNTH_NEON = "#ff2bd6"    # magenta neon
SYNTH_CYAN = "#11ffee"    # cyan neon
SYNTH_TEXT = "#e8e6ff"    # light text


class SynthwaveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthwave Cam → LM Studio (Text + Image)")
        self.root.configure(bg=SYNTH_BG)

        # OpenAI client pointing to LM Studio
        self.client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

        # Video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not access the webcam.")
        self.current_frame_bgr = None
        self.captured_image_pil = None

        # --- Styles (synthwave theme for ttk) ---
        self._build_styles()

        # --- Layout containers ---
        main = ttk.Frame(self.root, padding=10, style="Synth.TFrame")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Left: video + controls
        left = ttk.Frame(main, padding=8, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # Right: prompt + model + output
        right = ttk.Frame(main, padding=8, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1)

        # --- Left: Video feed ---
        self.video_label = ttk.Label(left, text="Webcam initializing...", anchor="center", style="Neon.TLabel")
        self.video_label.grid(row=0, column=0, columnspan=2, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        capture_btn = ttk.Button(left, text="⦿ Capture Frame", style="Neon.TButton", command=self.capture_frame)
        capture_btn.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        clear_btn = ttk.Button(left, text="✖ Clear Capture", style="NeonAlt.TButton", command=self.clear_capture)
        clear_btn.grid(row=1, column=1, sticky="ew", pady=(8, 0))

        # Captured preview
        self.captured_label = ttk.Label(left, text="(No capture yet)", anchor="center", style="Subtle.TLabel")
        self.captured_label.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        left.rowconfigure(2, weight=1)

        # --- Right: Model, prompt, send ---
        # Model row
        model_row = ttk.Frame(right, style="Panel.TFrame")
        model_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(model_row, text="Model:", style="NeonSmall.TLabel").grid(row=0, column=0, padx=(0, 6))
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_row, textvariable=self.model_var, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky="ew")
        model_row.columnconfigure(1, weight=1)
        refresh_btn = ttk.Button(model_row, text="↻ Refresh Models", style="NeonAlt.TButton", command=self.refresh_models)
        refresh_btn.grid(row=0, column=2, padx=(6, 0))

        # Prompt box
        ttk.Label(right, text="Your prompt:", style="NeonSmall.TLabel").grid(row=1, column=0, sticky="w")
        self.prompt_text = tk.Text(right, height=8, wrap="word", bg=SYNTH_PANEL, fg=SYNTH_TEXT,
                                   insertbackground=SYNTH_CYAN, relief="flat")
        self.prompt_text.grid(row=2, column=0, sticky="nsew", pady=(4, 6))
        right.rowconfigure(2, weight=1)
        # Buttons row
        btn_row = ttk.Frame(right, style="Panel.TFrame")
        btn_row.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.send_btn = ttk.Button(btn_row, text="🚀 Send to LLM", style="Neon.TButton", command=self.send_to_llm)
        self.send_btn.grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_row, textvariable=self.status_var, style="Subtle.TLabel").grid(row=0, column=1, sticky="e")
        btn_row.columnconfigure(1, weight=1)

        # Output box
        ttk.Label(right, text="Model response:", style="NeonSmall.TLabel").grid(row=4, column=0, sticky="w")
        self.output_text = tk.Text(right, height=12, wrap="word", bg=SYNTH_PANEL, fg=SYNTH_TEXT,
                                   insertbackground=SYNTH_CYAN, relief="flat")
        self.output_text.grid(row=5, column=0, sticky="nsew")
        right.rowconfigure(5, weight=1)

        # Start webcam loop
        self._video_loop()

        # Fetch models initially
        self.refresh_models()

        # Close cleanup
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------- Styles -------------------
    def _build_styles(self):
        style = ttk.Style()
        # Use default theme as base
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Synth.TFrame", background=SYNTH_BG)
        style.configure("Panel.TFrame", background=SYNTH_PANEL)

        style.configure("Neon.TLabel", background=SYNTH_PANEL, foreground=SYNTH_CYAN, font=("Montserrat", 12, "bold"))
        style.configure("NeonSmall.TLabel", background=SYNTH_PANEL, foreground=SYNTH_CYAN, font=("Montserrat", 10, "bold"))
        style.configure("Subtle.TLabel", background=SYNTH_PANEL, foreground=SYNTH_TEXT, font=("Inter", 10))

        style.configure("Neon.TButton", background=SYNTH_NEON, foreground="#120020")
        style.map("Neon.TButton",
                  background=[("active", "#ff5be0"), ("disabled", "#802a79")],
                  foreground=[("disabled", "#201032")])
        style.configure("NeonAlt.TButton", background=SYNTH_CYAN, foreground="#102225")
        style.map("NeonAlt.TButton",
                  background=[("active", "#6ffff6"), ("disabled", "#2a6e6a")],
                  foreground=[("disabled", "#1d3d3a")])

        # Make ttk widgets blend in
        style.configure("TCombobox", fieldbackground=SYNTH_PANEL, background=SYNTH_PANEL, foreground=SYNTH_TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", SYNTH_PANEL)])
        style.configure("TLabel", background=SYNTH_PANEL, foreground=SYNTH_TEXT)

    # ------------------- Video Loop -------------------
    def _video_loop(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_bgr = frame
                # Convert BGR->RGB and display scaled
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((480, int(480 * img.height / img.width))) if img.width > 480 else img
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=imgtk, text="")
                self.video_label.image = imgtk
        self.root.after(30, self._video_loop)  # ~33 FPS cap

    def capture_frame(self):
        if self.current_frame_bgr is None:
            messagebox.showwarning("No Frame", "No video frame available to capture yet.")
            return
        # Freeze the current frame into a PIL image (RGB)
        frame_rgb = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2RGB)
        self.captured_image_pil = Image.fromarray(frame_rgb)
        # Show thumbnail
        preview = self.captured_image_pil.copy()
        preview.thumbnail((320, 320))
        imgtk = ImageTk.PhotoImage(preview)
        self.captured_label.configure(image=imgtk, text="")
        self.captured_label.image = imgtk
        self.status_var.set("Captured current frame.")

    def clear_capture(self):
        self.captured_image_pil = None
        self.captured_label.configure(image="", text="(No capture yet)")
        self.captured_label.image = None
        self.status_var.set("Cleared captured image.")

    # ------------------- LM Studio Integration -------------------
    def refresh_models(self):
        def worker():
            try:
                self.status_var.set("Fetching models from LM Studio…")
                resp = self.client.models.list()
                ids = [m.id for m in getattr(resp, "data", [])]
                if not ids:
                    raise RuntimeError("No models returned. Is LM Studio server running and a model loaded?")
                # Update dropdown on main thread
                self.root.after(0, self._update_models_ui, ids)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Model Error", f"Could not fetch models:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("Model fetch failed."))

        threading.Thread(target=worker, daemon=True).start()

    def _update_models_ui(self, model_ids):
        self.model_combo["values"] = model_ids
        # Select first by default, keep previous if still present
        current = self.model_var.get()
        if current in model_ids:
            self.model_combo.set(current)
        else:
            self.model_combo.set(model_ids[0])
        self.status_var.set("Models updated.")

    def send_to_llm(self):
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning("No Model", "Please select a model.")
            return

        user_text = self.prompt_text.get("1.0", "end").strip()
        if not user_text and self.captured_image_pil is None:
            messagebox.showwarning("Nothing to Send", "Write a prompt and/or capture an image first.")
            return

        # Disable send while running
        self.send_btn.config(state="disabled")
        self.status_var.set("Sending request…")

        def worker():
            try:
                content_items = []
                if user_text:
                    content_items.append({"type": "text", "text": user_text})

                if self.captured_image_pil is not None:
                    # Encode as PNG base64 data URL
                    buf = io.BytesIO()
                    self.captured_image_pil.save(buf, format="PNG")
                    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                    data_url = f"data:image/png;base64,{encoded}"
                    # LM Studio (OpenAI-compatible) commonly accepts image_url for input images
                    content_items.append({"type": "input_image", "image_url": data_url})

                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content_items}],
                    temperature=0.2,
                )

                text = ""
                try:
                    text = completion.choices[0].message.content or ""
                except Exception:
                    text = str(completion)

                self.root.after(0, lambda: self._set_output(text))
                self.root.after(0, lambda: self.status_var.set("Done."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Request Error", f"LLM request failed:\n{e}"))
                self.root.after(0, lambda: self.status_var.set("Failed."))
            finally:
                self.root.after(0, lambda: self.send_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _set_output(self, text):
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text)

    # ------------------- Cleanup -------------------
    def on_close(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()

    # Make window reasonably compact but roomy for text
    root.geometry("1100x720")  # width x height

    app = SynthwaveApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

