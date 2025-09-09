import tkinter as tk
from tkinter import ttk, scrolledtext
from src.path_helper import resource_path
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
        
        # Configurar icono
        icon_path = resource_path("img/icon.ico")
        if os.path.exists(icon_path):
            self.window.iconbitmap(icon_path)
        
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
            text="Sistema Inteligente de Detección de Infracciones v1.0",
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
        """Cargar contenido actualizado del manual"""
        content = """
📄 DESCRIPCIÓN

InfractiVision es un software de escritorio diseñado para detectar y registrar infracciones de tránsito, con énfasis en el cruce indebido de luz roja en intersecciones críticas.

El sistema combina:
• Visión computacional avanzada
• Reconocimiento automático de matrículas (OCR)
• Semáforo virtual sincronizado
• Panel visual de gestión de videos
• Almacenamiento automático en Google Cloud
• Análisis de rendimiento en tiempo real

⚙️ REQUISITOS DEL SISTEMA

✅ SISTEMA OPERATIVO:
   • Windows 10 de 64-bit o superior
   • Windows 11 (recomendado)

✅ HARDWARE MÍNIMO:
   • Procesador: Intel i5 o AMD Ryzen 5 (o equivalente)
   • RAM: 8GB mínimo, 16GB recomendado
   • GPU: NVIDIA con soporte CUDA (opcional pero recomendado)
   • Espacio en disco: 2GB libres para instalación + espacio para videos

✅ CONECTIVIDAD:
   • Conexión a Internet estable (para migración automática a la nube)
   • Sin Internet: Funciona en modo local con limitaciones

✅ FORMATOS DE VIDEO SOPORTADOS:
   • MP4 (recomendado), AVI, MOV, MKV, WMV, FLV
   • Resolución: 480p mínimo, 1080p recomendado
   • Framerate: 15-30 FPS

🔧 DETECCIÓN AUTOMÁTICA DE HARDWARE

El sistema se adapta automáticamente:
• 🚀 GPU NVIDIA con CUDA: Resolución 640px, procesamiento acelerado
• 💻 Solo CPU: Resolución 320px optimizada, skip frames inteligente
• ⚡ Optimización dinámica según rendimiento detectado

🌐 VERIFICACIÓN DE CONEXIÓN

• Con Internet: Migración automática a Google Cloud/Firestore
• Sin Internet: Almacenamiento local + aviso de reconexión

🚀 GUÍA DE USO RÁPIDO

1️⃣ CONFIGURACIÓN DE VIDEOS
   • Haga clic en "CONFIGURACIÓN DE VIDEOS"
   • Seleccione un video de la galería visual
   • Si no está configurado: use "⚙️ Configurar"
   • Si ya está configurado: aparecerá "🔒 Bloqueado" y "✏️ Editar Config"

2️⃣ CONFIGURACIÓN COMPLETA
   • Avenida/Ubicación del video
   • Tiempos del semáforo (Verde, Amarillo, Rojo)
   • Área restrictiva (polígono donde detectar infracciones)

3️⃣ PROCESAMIENTO DE INFRACCIONES
   • Clic en "INICIAR PROCESAMIENTO DE INFRACCIONES"
   • El sistema verificará configuración automáticamente
   • Se abrirá ventana con semáforo sincronizado
   • Análisis en tiempo real con indicadores GPU/CPU

4️⃣ GESTIÓN DE INFRACCIONES
   • Visualización de infracciones detectadas
   • Exportación a JSON, CSV, Excel
   • Indicadores de rendimiento (TI, TR, IR)
   • Galería de placas detectadas

🎯 FUNCIONALIDADES AVANZADAS

📹 SELECTOR VISUAL DE VIDEOS
• Miniaturas automáticas de videos
• Estado de configuración en tiempo real (🔒 Configurado, ⚙️ Sin configurar)
• Botones inteligentes según estado
• Acciones: Seleccionar, Configurar, Limpiar, Eliminar, Importar

🚦 SEMÁFORO SINCRONIZADO INTELIGENTE
• Configuración personalizada por intersección
• Tiempos ajustables: Verde, Amarillo, Rojo
• Sincronización perfecta entre ventanas
• Indicador visual de tiempo restante
• ⚠️ Adaptación automática para videos nocturnos

🌙 DETECCIÓN NOCTURNA AVANZADA
• Análisis automático de luminosidad del video
• Ventana de calibración específica para condiciones nocturnas
• Configuración optimizada para baja iluminación
• ⚠️ Limitaciones: Menor precisión en detección nocturna

⚡ MOTOR DE DETECCIÓN MULTINIVEL
• YOLO v8: Detección vehicular de alta precisión
• EasyOCR: Reconocimiento automático de placas (ANPR)
• Área restrictiva configurable (polígono personalizable)
• Filtros anti-ruido para placas inválidas

🗃️ GESTIÓN ACUMULATIVA DE INFRACCIONES
• Lista tipo stack/pila (infracciones nuevas al principio)
• Exportación múltiple: JSON, CSV, Excel
• Filtrado por fechas y ubicaciones
• Galería visual de evidencias (placas + vehículos)

☁️ MIGRACIÓN AUTOMÁTICA E INDICADORES
• Subida automática a Google Cloud/Firestore
• Generación de indicadores de rendimiento (TI, TR, IR)
• Comparativas antes/después del software
• Identificación única por dispositivo

⚠️ LIMITACIONES CONOCIDAS Y CONSEJOS

🌙 VIDEOS NOCTURNOS:
• Menor precisión en detección de placas
• Requiere calibración específica
• Se recomienda iluminación mínima en la intersección

🎯 DETECCIÓN ÓPTIMA:
• Videos diurnos con buena iluminación
• Cámaras estáticas (sin movimiento)
• Ángulo frontal/semi-frontal a las placas
• Resolución mínima 720p para mejor OCR

📊 RENDIMIENTO:
• Sin límite fijo de infracciones detectadas por video
• Detección basada en área restrictiva configurada
• Precisión depende de calidad del video y calibración

🔧 SOLUCIÓN DE PROBLEMAS:
• Si no detecta infracciones: Verificar área restrictiva y tiempos del semáforo
• Si placas ilegibles: Mejorar resolución del video o ángulo de cámara
• Si errores de exportación: Verificar permisos de escritura en carpeta destino

🔒 SEGURIDAD Y AUTENTICACIÓN

Cada usuario y dispositivo tiene un identificador único generado automáticamente, 
asegurando control de registros sin necesidad de inicio de sesión manual.

📧 INFORMACIÓN DEL PROYECTO

Autor: Abel Jesús Moya Acosta
Email: amoyaa2@upao.edu.pe
Institución: Universidad Privada Antenor Orrego (UPAO)
Proyecto de Tesis 2025

Versión: 1.0
Sistema desarrollado con:
• Python + OpenCV + PyTorch
• YOLOv8 para detección vehicular
• EasyOCR para reconocimiento de placas
• Google Cloud Platform para almacenamiento
• Tkinter para interfaz gráfica

🎉 ¡GRACIAS POR USAR INFRACTIVISION!
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
