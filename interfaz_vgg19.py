"""Interfaz de escritorio para el proyecto VGG19 de colonoscopia.

Colocar este archivo en la raíz de CancerColon_IA.
Requiere: tensorflow/keras, numpy, Pillow, tkinter.
"""
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import cv2
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelos" / "vgg19_colon.keras"
CHART_DIR = BASE_DIR / "resultados" / "graficas"
LOGO_PATH = BASE_DIR / "recursos" / "logo.png"

NAVY = "#AD3333"        # color principal
NAVY_DARK = "#8E2525"   # versión más oscura para footer y hover
TEAL = "#C44A4A"        # acento secundario
BG = "#FAF5F5"          # fondo general claro con tono cálido
WHITE = "#FFFFFF"
INK = "#4A2323"         # texto principal
MUTED = "#7A5A5A"       # texto secundario
BORDER = "#E7D4D4"      # bordes suaves
FONT = "Segoe UI"
CHARTS = [
    ("01_accuracy.png", "Exactitud del entrenamiento"),
    ("02_loss.png", "Pérdida del entrenamiento"),
    ("03_metricas_clasificacion.png", "Métricas por categoría"),
    ("04_aciertos_errores.png", "Aciertos y errores"),
    ("05_distribucion_puntuaciones.png", "Distribución de puntuaciones"),
    ("06_curva_roc.png", "Curva ROC"),
    ("07_curva_precision_recall.png", "Curva Precision-Recall"),
    ("08_matriz_confusion.png", "Matriz de confusión"),
    ("09_puntuaciones_ampliadas.png", "Puntuaciones cercanas al umbral"),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análisis de colonoscopias | VGG19")
        self.geometry("1160x760")
        self.minsize(850, 580)
        self.configure(bg=BG)
        self._images = {}
        self.selected_file = None
        self.model = None
        self.model_error = None
        self.running = False

        self.header = tk.Frame(self, bg=NAVY, height=88)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)
        self._make_header()
        self.main = tk.Frame(self, bg=BG)
        self.main.pack(fill="both", expand=True)
        self.footer = tk.Label(
            self, text="Proyecto experimental de investigación · No apto para diagnóstico clínico",
            bg=NAVY_DARK, fg="#F8EAEA", font=(FONT, 10), pady=10
        )
        self.footer.pack(fill="x")
        self.show_home()

    def _make_header(self):
        left = tk.Frame(self.header, bg=NAVY)
        left.pack(side="left", padx=28, pady=12)
        if LOGO_PATH.is_file():
            try:
                img = Image.open(LOGO_PATH).convert("RGBA")
                img.thumbnail((100, 60), Image.Resampling.LANCZOS)
                self._images["logo"] = ImageTk.PhotoImage(img)
                tk.Label(left, image=self._images["logo"], bg=NAVY).pack(side="left", padx=(0, 15))
            except Exception:
                self._logo_placeholder(left)
        else:
            self._logo_placeholder(left)
        info = tk.Frame(left, bg=NAVY)
        info.pack(side="left")
        tk.Label(info, text="ANÁLISIS DE COLONOSCOPIAS", font=(FONT, 18, "bold"),
                 bg=NAVY, fg=WHITE).pack(anchor="w")
        tk.Label(info, text="Clasificación experimental de imágenes · Arquitectura VGG19",
                 font=(FONT, 10), bg=NAVY, fg="#F1D6D6").pack(anchor="w")
        tk.Button(self.header, text="⌂  Inicio", command=self.show_home, bg="#C04A4A", fg=WHITE,
          activebackground="#8E2525", activeforeground=WHITE, relief="flat",
                  cursor="hand2", font=(FONT, 11), padx=16, pady=8).pack(side="right", padx=26)

    def _logo_placeholder(self, parent):
        tk.Label(parent, text="LOGO", bg=WHITE, fg=NAVY, font=(FONT, 12, "bold"),
                 width=9, height=3).pack(side="left", padx=(0, 15))

    def clear(self):
        for child in self.main.winfo_children():
            child.destroy()

    def heading(self, parent, title, subtitle):
        tk.Label(parent, text=title, font=(FONT, 23, "bold"), fg=INK, bg=BG).pack(anchor="w")
        tk.Label(parent, text=subtitle, font=(FONT, 11), fg=MUTED, bg=BG,
                 wraplength=800, justify="left").pack(anchor="w", pady=(4, 20))

    def button(self, parent, text, command, primary=True, **kwargs):
        return tk.Button(parent, text=text, command=command,
                         bg=NAVY if primary else WHITE, fg=WHITE if primary else NAVY,
                         activebackground=NAVY_DARK if primary else "#F4E6E6",
                         activeforeground=WHITE if primary else NAVY,
                         font=(FONT, 12, "bold"), relief="flat", cursor="hand2",
                         padx=18, pady=12, **kwargs)

    def show_home(self):
        self.clear()
        outer = tk.Frame(self.main, bg=BG)
        outer.pack(expand=True)
        tk.Label(outer, text="Bienvenido al sistema de análisis", font=(FONT, 27, "bold"),
                 fg=INK, bg=BG).pack(pady=(0, 10))
        tk.Label(outer, text="Seleccione una opción para explorar los resultados o analizar una imagen.",
                 font=(FONT, 12), fg=MUTED, bg=BG).pack(pady=(0, 34))
        cards = tk.Frame(outer, bg=BG)
        cards.pack()
        for title, subtitle, symbol, action in [
            ("GRÁFICAS", "Consulte las estadísticas del entrenamiento y evaluación", "▥", self.show_charts),
            ("PREDICCIÓN", "Seleccione una imagen de colonoscopia para clasificarla", "◉", self.show_prediction),
        ]:
            card = tk.Frame(cards, bg=WHITE, highlightbackground=BORDER, highlightthickness=1,
                            width=340, height=260)
            card.pack(side="left", padx=12)
            card.pack_propagate(False)
            tk.Label(card, text=symbol, font=(FONT, 48), fg=TEAL, bg=WHITE).pack(pady=(24, 8))
            tk.Label(card, text=title, font=(FONT, 16, "bold"), fg=INK, bg=WHITE).pack()
            tk.Label(card, text=subtitle, font=(FONT, 10), fg=MUTED, bg=WHITE,
                     wraplength=280, justify="center").pack(pady=12)
            self.button(card, "Abrir", action).pack()

    def show_charts(self):
        self.clear()
        frame = tk.Frame(self.main, bg=BG)
        frame.pack(fill="both", expand=True, padx=28, pady=22)
        self.heading(frame, "Gráficas y estadísticas", "Seleccione una gráfica para visualizarla en detalle. Se utilizan las imágenes ya generadas por el proyecto.")
        body = tk.Frame(frame, bg=BG)
        body.pack(fill="both", expand=True)
        menu = tk.Frame(body, bg=WHITE, width=290, highlightbackground=BORDER, highlightthickness=1)
        menu.pack(side="left", fill="y", padx=(0, 16))
        menu.pack_propagate(False)
        # Menú vertical desplazable, incluso cuando la ventana es pequeña.
        canvas = tk.Canvas(menu, bg=WHITE, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(menu, orient="vertical", command=canvas.yview)
        entries = tk.Frame(canvas, bg=WHITE)
        window_id = canvas.create_window((0, 0), window=entries, anchor="nw")

        def ajustar_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def ajustar_ancho(event):
            canvas.itemconfigure(window_id, width=event.width)

        def desplazar(event):
            # En Windows el delta suele ser múltiplo de 120.
            if event.delta:
                pasos = -int(event.delta / 120)
                if pasos == 0:
                    pasos = -1 if event.delta > 0 else 1
                canvas.yview_scroll(pasos, "units")
            return "break"

        entries.bind("<Configure>", ajustar_region)
        canvas.bind("<Configure>", ajustar_ancho)
        canvas.configure(yscrollcommand=scrollbar.set, yscrollincrement=24)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.bind("<MouseWheel>", desplazar)
        entries.bind("<MouseWheel>", desplazar)
        viewer = tk.Frame(body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        viewer.pack(side="left", fill="both", expand=True)
        self.chart_label = tk.Label(viewer, text="Seleccione una gráfica", bg=WHITE, fg=MUTED,
                                    font=(FONT, 14))
        self.chart_label.pack(fill="both", expand=True, padx=16, pady=12)
        for filename, title in CHARTS:
            button = tk.Button(
                entries, text=title, anchor="w", wraplength=230,
                justify="left", font=(FONT, 11), relief="flat",
                cursor="hand2", bg=WHITE, fg=INK,
                activebackground="#F3E3E3", padx=12, pady=14,
                command=lambda f=filename: self.display_chart(f)
            )
            button.pack(fill="x")
            # La rueda también funciona al pasar el cursor por un botón.
            button.bind("<MouseWheel>", desplazar)
        ajustar_region()
        self.display_chart(CHARTS[0][0])
        self._chart_viewer = viewer

    def display_chart(self, filename):
        path = CHART_DIR / filename
        if not path.is_file():
            self.chart_label.configure(image="", text=f"No se encontró la gráfica:\n{path}\n\nEjecute graficas_estadisticas.py primero.")
            return
        try:
            img = Image.open(path).convert("RGB")
            # Measure current available space (can be small during initial rendering)
            self.update_idletasks()
            width = max(460, self.chart_label.winfo_width() - 24)
            height = max(380, self.chart_label.winfo_height() - 24)
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._images["chart"] = photo
            self.chart_label.configure(image=photo, text="")
        except Exception as exc:
            self.chart_label.configure(image="", text=f"No se pudo abrir la gráfica: {exc}")

    def show_prediction(self):
        self.clear()
        frame = tk.Frame(self.main, bg=BG)
        frame.pack(fill="both", expand=True, padx=28, pady=22)
        self.heading(frame, "Predicción de imágenes", "Seleccione una imagen JPG, JPEG o PNG. La aplicación mostrará las puntuaciones de las categorías POLY y NORM.")
        body = tk.Frame(frame, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1,
                        width=515)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = tk.Frame(body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1,
                         width=400)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="Imagen seleccionada", font=(FONT, 15, "bold"), bg=WHITE,
                 fg=INK).pack(anchor="w", padx=20, pady=(20, 12))
        self.preview_label = tk.Label(left, text="Todavía no hay una imagen seleccionada",
                                      bg="#EAF0F5", fg=MUTED, font=(FONT, 11),
                                      height=17, wraplength=350)
        self.preview_label.pack(fill="both", expand=True, padx=20, pady=(0, 14))
        buttons = tk.Frame(left, bg=WHITE)
        buttons.pack(pady=(0, 20))
        self.button(buttons, "Seleccionar imagen", self.choose_image).pack(side="left", padx=5)
        self.predict_button = self.button(buttons, "Realizar predicción", self.predict_image)
        self.predict_button.pack(side="left", padx=5)

        tk.Label(right, text="Resultado del modelo", font=(FONT, 15, "bold"),
                 bg=WHITE, fg=INK).pack(anchor="w", padx=22, pady=(20, 16))
        self.result_title = tk.Label(right, text="Esperando una imagen", bg=WHITE,
                                     fg=NAVY, font=(FONT, 19, "bold"), wraplength=340)
        self.result_title.pack(anchor="w", padx=22, pady=(0, 12))
        self.info_label = tk.Label(right, text="Seleccione una fotografía para comenzar.",
                                   bg=WHITE, fg=MUTED, font=(FONT, 11), wraplength=340, justify="left")
        self.info_label.pack(anchor="w", padx=22, pady=(0, 26))
        self.score_bars = {}
        for class_name in ("POLY", "NORM"):
            group = tk.Frame(right, bg=WHITE)
            group.pack(fill="x", padx=22, pady=(0, 20))
            row = tk.Frame(group, bg=WHITE)
            row.pack(fill="x", pady=(0, 6))
            tk.Label(row, text=class_name, font=(FONT, 11, "bold"), bg=WHITE, fg=INK).pack(side="left")
            value = tk.Label(row, text="--", font=(FONT, 11), bg=WHITE, fg=INK)
            value.pack(side="right")
            bar = ttk.Progressbar(group, maximum=100, value=0, mode="determinate")
            bar.pack(fill="x")
            self.score_bars[class_name] = (value, bar)
        tk.Label(right, text="Las puntuaciones del modelo no constituyen un diagnóstico médico.",
                 bg=WHITE, fg=MUTED, font=(FONT, 10), justify="left",
                 wraplength=340).pack(side="bottom", anchor="w", padx=22, pady=22)
        if self.selected_file is not None and self.selected_file.is_file():
            self._show_preview(self.selected_file)

    def _show_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((430, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._images["preview"] = photo
            self.preview_label.configure(image=photo, text="")
        except Exception as exc:
            self.preview_label.configure(image="", text=f"No se pudo mostrar la imagen: {exc}")

    def choose_image(self):
        selection = filedialog.askopenfilename(title="Seleccionar imagen de colonoscopia",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")])
        if not selection:
            return
        self.selected_file = Path(selection)
        self._show_preview(self.selected_file)
        self.result_title.configure(text="Lista para analizar", fg=NAVY)
        self.info_label.configure(text=self.selected_file.name)
        for value, bar in self.score_bars.values():
            value.configure(text="--")
            bar.configure(value=0)

    def predict_image(self):
        if self.running:
            return
        if self.selected_file is None:
            messagebox.showinfo("Seleccione una imagen", "Primero seleccione una imagen para analizar.")
            return
        if not MODEL_PATH.is_file():
            messagebox.showerror("Modelo no encontrado", f"No se encontró el modelo:\n{MODEL_PATH}")
            return
        self.running = True
        self.predict_button.configure(state="disabled")
        self.result_title.configure(text="Analizando imagen...", fg=NAVY)
        self.info_label.configure(text="Cargando y ejecutando VGG19. Espere un momento.")
        path = self.selected_file
        threading.Thread(target=self._predict_worker, args=(path,), daemon=True).start()

    def _predict_worker(self, path):
        try:
            # Importar Keras en el hilo para evitar bloquear el inicio de la interfaz.
            from keras.models import load_model
            from keras.applications.vgg19 import preprocess_input
            if self.model is None:
                self.model = load_model(MODEL_PATH, compile=False)
            # El entrenamiento lee las imágenes con OpenCV, las transforma a RGB,
            # redimensiona y aplica preprocess_input de VGG19.
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("No se pudo leer la imagen seleccionada")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (150, 150))
            data = np.asarray(image, dtype=np.float32)
            batch = preprocess_input(np.expand_dims(data, axis=0))
            scores = self.model.predict(batch, verbose=0)[0]
            poly, norm = float(scores[0]), float(scores[1])
            self.after(0, lambda: self._show_result(path, poly, norm))
        except Exception as exc:
            error = str(exc)
            self.after(0, lambda: self._show_error(error))

    def _show_result(self, path, poly, norm):
        self.running = False
        self.predict_button.configure(state="normal")
        if path != self.selected_file:
            return
        category = "POLY" if poly > norm else "NORM"
        self.result_title.configure(text=f"Clasificación: {category}", fg=TEAL if category == "POLY" else NAVY)
        self.info_label.configure(text=f"Imagen: {path.name}\nCategoría con mayor puntuación: {category}")
        for name, score in (("POLY", poly), ("NORM", norm)):
            label, bar = self.score_bars[name]
            label.configure(text=f"{score * 100:.2f} %")
            bar.configure(value=score * 100)

    def _show_error(self, error):
        self.running = False
        self.predict_button.configure(state="normal")
        self.result_title.configure(text="No se pudo completar la predicción", fg="#9C3939")
        self.info_label.configure(text=error)
        messagebox.showerror("Error de predicción", error)


if __name__ == "__main__":
    App().mainloop()
