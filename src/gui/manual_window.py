import tkinter as tk
from tkinter import ttk, scrolledtext
from src.path_helper import resource_path
from src.core.utils.icon import set_window_icon
import os

class ManualWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = None
        self.create_manual_window()

    def create_manual_window(self):
        """Crear ventana del manual estilizada"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("📖 Manual de Usuario - InfractiVision")
        self.window.geometry("900x700")
        self.window.configure(bg='white')
        self.window.resizable(True, True)

        set_window_icon(self.window)

        # Hacer ventana modal
        self.window.transient(self.parent)
        self.window.grab_set()

        self.create_content()

    def create_content(self):
        """Crear contenido del manual"""
        # Header con gradiente
        header_frame = tk.Frame(self.window, bg='#2c3e50', height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)

        # Título principal
        title_label = tk.Label(
            header_frame,
            text="📖 InfractiVision - Manual de Usuario",
            font=("Arial", 18, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(expand=True)

        # Subtítulo
        subtitle_label = tk.Label(
            header_frame,
            text="Sistema Inteligente de Deteccion de Infracciones por Cruce en Rojo",
            font=("Arial", 10),
            bg='#2c3e50',
            fg='#bdc3c7'
        )
        subtitle_label.pack()

        # Contenedor principal con scroll
        main_frame = tk.Frame(self.window, bg='white')
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Text widget con scroll
        self.text_widget = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg='white',
            fg='#2c3e50',
            selectbackground='#3498db',
            selectforeground='white',
            padx=15,
            pady=15
        )
        self.text_widget.pack(fill="both", expand=True)

        # Cargar contenido del manual
        self.load_manual_content()

        # Frame de botones
        button_frame = tk.Frame(self.window, bg='white')
        button_frame.pack(fill="x", padx=20, pady=(0, 20))

        close_btn = tk.Button(
            button_frame,
            text="✓ Cerrar Manual",
            command=self.close_manual,
            bg="#27ae60",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=30,
            pady=10,
            cursor="hand2"
        )
        close_btn.pack(side="right")

    def load_manual_content(self):
        """Cargar contenido actualizado del manual (sincronizado con README.md)"""
        content = """
DESCRIPCION

InfractiVision es un software de escritorio que detecta vehiculos que cruzan
en rojo a partir de video grabado. El flujo es offline por video: eliges un
archivo, configuras la zona y los tiempos del semaforo, procesas, validas las
placas y el sistema guarda metricas localmente y las migra a la nube.

Proyecto de tesis UPAO 2025 - Abel Jesus Moya Acosta.


1. QUE PUEDE HACER EL SOFTWARE
- Deteccion en rojo: cruces de un poligono configurable solo en fase roja
  (mas un pequeno pre-rojo) del ciclo G -> Y -> R simulado.
- Tracking: ID por vehiculo (centroide / DeepSORT) para no duplicar.
- Placas: YOLO dedicado + mejor crop por infractor (scoring de calidad:
  contraste, bordes, nitidez, tamano).
- OCR principal LPRNet Peru (LPRNet_Peru_MASTER_FINAL.pth) con contexto
  Trujillo y validacion SIIV MTC. Alternativos PaddleOCR/EasyOCR por env.
- Validacion en la nube: cada crop se valida contra Plate Recognizer API
  (regions=pe) con revision humana (check Validar).
- Indicadores NID / NIE / TI / TR. NID = validadas con placa;
  NIE = pendientes sin placa o no validadas.
- Persistencia local SQLite (data/infractions.sqlite).
- Migracion a Firebase Firestore (infractivision-e8c03, migraciones/{uuid}).
- Exportacion CSV / Excel / PDF / JSON.
- Deteccion nocturna (nombre de video o brillo < 60).
- Gestion de videoteca con 5 videos demo descargables.


2. PARTES DEL SOFTWARE
- Welcome (inicio): 3 botones - Manual de Usuario, Foto Rojo,
  Gestion de Infracciones.
- Selector de videos: galeria con miniatura, duracion/resolucion/tamano y
  estado (Configurado / Sin configurar). Acciones: Seleccionar, Configurar,
  Limpiar, Eliminar, Importar, Actualizar, Descargar Demo.
- Configuracion de zona: poligono de la interseccion + margen de peligro
  (default 80 px). Se guarda en la BD (tabla video_configs, fuente única).
- Configuracion de semaforo + avenida: tiempos G/Y/R + pre-rojo (0.5 s) +
  green_skip_rate (60 frames) + nombre de avenida.
- Reproductor: play/pause, overlay SEMAFORO EN ROJO, timer HH:mm:ss y
  estado G/Y/R, beeps.
- Preprocesamiento: dialogo de progreso que lanza el OfficialVideoProcessor
  en hilo worker. Al terminar congela semaforo/timer.
- Validacion de placas (PlateReviewWindow): revision secuencial crop por crop.
  Botones: Reintentar actual, Exportar validados, Completado (guarda en
  SQLite + calcula TI/TR + migra a Firestore).
- Gestion de Infracciones: tarjetas a color (verde NID / ambar NIE),
  paginacion de 10 con Cargar mas, filtros por fecha/placa, exportar
  CSV/Excel/PDF, eliminar, historial de migraciones.


3. COMO SE USA (PASO A PASO)

Paso 0 - Abrir: python main.py o doble clic en InfractiVision.exe.
  Video ideal: MP4, 720p minimo (1080p ideal), 15-30 FPS, camara estatica
  con vista frontal a las placas.

Paso 1 - Pulsa "Foto Rojo" en el inicio.

Paso 2 - Elige video en el selector. Si dice Sin configurar, pulsa
  Configurar: dibuja el poligono sobre la interseccion (clic por vertice),
  pon avenida (ej. Av. Condorcanqui - Trujillo) y tiempos Verde / Amarillo /
  Rojo en segundos (ej. 12 / 2 / 10). Guarda.

Paso 3 - Reproduce y verifica que el banner SEMAFORO EN ROJO y el timer
  cuadran con el video. Si no, vuelve a Configurar.

Paso 4 - Pulsa INICIAR PROCESAMIENTO DE INFRACCIONES. Espera la barra de
  progreso (corre en background, no cierres la ventana).

Paso 5 - Valida placas en PlateReviewWindow:
  1) Espera el texto de Plate Recognizer (Confianza: 0.xx, ~2 s por placa).
  2) Corrige la placa a mano si el OCR se equivoco (formato Peru ABC-123).
  3) Marca Validar solo en las correctas (cuentan como NID).
  4) Reintentar actual si fallo la red; Exportar validados para CSV rapido.
  5) Pulsa Completado para guardar + migrar.

Paso 6 - Revisa en Gestion de Infracciones: filtra por fecha o placa,
  Cargar mas (10 en 10), exporta a CSV/Excel/PDF o elimina. La pestana de
  migraciones muestra el uuid subido a Firestore.


4. INDICADORES
- NID: evidencias validadas con placa.
- NIE: pendientes sin placa + no validados.
- TI = NID / (NID+NIE) * 100 (%).
- TR = duracion del video (min) / NID (min por infraccion).


5. COMPILAR BUILD (desarrolladores)
- Requisito: Python 3.10 64-bit. Verifica: struct.calcsize('P')*8 == 64.
  Prohibido opencv-python-headless en prod (solo tests).
- Canonico CUDA: pip install -r requirements.txt, luego
  python scripts/build_online.py --variant cuda
  (salida dist/InfractiVision/ = ONEDIR, arranque < 1.5 s).
- Con ZIP para Releases: ... --variant cuda --zip.
- Legacy CPU/macOS: pip install -r requirements-cpu.txt, luego
  python scripts/build_online.py --variant cpu --zip.
- ONEFILE portable: python scripts/build_online.py --variant cuda --onefile.
- Setup Windows: iscc installer/win/online.iss
  (genera dist/InfractiVision-Setup-Online.exe, requiere Inno Setup 6).
- Linux: bash installer/linux/install.sh. Mac: bash installer/mac/install.sh.
- Verificar: python scripts/verify_installer.py y ci_smoke_test.py.
- Release CI: git tag v2.1.0 + git push origin v2.1.0 (release.yml compila
  y publica el Setup solo). Secrets: FIREBASE_SA_JSON y
  PLATE_RECOGNIZER_TOKEN.


6. SI ALGO SALE MAL
- No detecta: revisa poligono (que cubra el cruce) y tiempos G/Y/R.
- Placas mal leidas: usa 1080p diurno si puedes; SmartPlateCorrector corrige
  0-O / 1-I / 8-B pero no el blur extremo.
- Confianza eternamente vacia: falta PLATE_RECOGNIZER_API_TOKEN en .env o
  no hay internet (queda como NIE, reintenta luego).
- No migra a Firestore: falta JSON Service Account o internet (queda
  pendiente en tabla migrations).
- Error al exportar: carpeta sin permiso, exporta a Documentos/.


Autor: Abel Jesus Moya Acosta - amoyaa2@upao.edu.pe
Universidad Privada Antenor Orrego (UPAO) - Tesis 2025/2026.
Stack: Python 3.10, OpenCV 4.9, YOLOv8, PyTorch, LPRNet Peru,
Plate Recognizer, Firestore, Tkinter.

GRACIAS POR USAR INFRACTIVISION!
"""

        # Insertar contenido con formato
        self.text_widget.insert("1.0", content)

        # Configurar tags para formato
        self.text_widget.tag_configure("header", font=("Arial", 14, "bold"), foreground="#2c3e50")
        self.text_widget.tag_configure("emoji", font=("Segoe UI Emoji", 12))

        # Hacer el texto de solo lectura
        self.text_widget.configure(state="disabled")

    def close_manual(self):
        """Cerrar ventana del manual"""
        self.window.destroy()

def show_manual(parent):
    """Función utilitaria para mostrar el manual"""
    manual = ManualWindow(parent)
    return manual.window
