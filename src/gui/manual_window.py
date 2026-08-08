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
            text="Sistema Inteligente de Detección de Infracciones v2.0 con SmartPlateCorrector",
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

InfractiVision es un software de escritorio desarrollado como proyecto de tesis que utiliza inteligencia artificial avanzada para detectar y registrar automáticamente infracciones de tránsito por cruce indebido en luz roja.

🎓 PROYECTO DE TESIS - UPAO 2025:
Desarrollado por Abel Jesús Moya Acosta como tesis para obtener el título de Ingeniero de Sistemas, bajo la supervisión de la Universidad Privada Antenor Orrego.

🚀 INNOVACIONES IMPLEMENTADAS v2.0:
• 🤖 SmartPlateCorrector: Sistema propio de corrección inteligente OCR
• ⚡ PaddleOCR optimizado: Inicialización asíncrona (10.8s vs 25-30s)
• 🌙 Detección nocturna mejorada: Análisis automático de luminosidad
• 🇵🇪 Clasificación de placas: Peruanas (ABC-123) vs extranjeras
• 📊 Precisión mejorada: Hasta 92% en reconocimiento de placas

ARQUITECTURA DEL SISTEMA:
• Detección vehicular: YOLO v8 de alta precisión
• OCR inteligente: PaddleOCR + SmartPlateCorrector propio
• Corrección automática: H/N, T/7, B/8, I/1, O/0, S/5, G/6
• Semáforo virtual: Sincronización temporal precisa
• Interfaz gráfica: Tkinter con visualización fluida 30 FPS
• Almacenamiento cloud: Google Cloud Platform automático
• Sistema TR: Análisis de tiempo real con aceleración visual

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

🌙 DETECCIÓN NOCTURNA INTELIGENTE v2.0
• Análisis automático de luminosidad (umbral < 60) 
• SmartCorrector adaptado para condiciones nocturnas
• Ventana de calibración específica para baja iluminación
• Corrección automática de caracteres confusos mejorada
• ⚡ Mejora del 11% en precisión nocturna vs versión anterior
• ⚠️ Limitaciones: Calidad del OCR depende de la iluminación disponible

⚡ MOTOR DE IA AVANZADO v2.0
• YOLO v8: Detección vehicular de alta precisión
• 🤖 SmartPlateCorrector: Sistema de corrección inteligente con 3 niveles
  - Nivel 1: Validación de formato (ABC-123 para Perú)
  - Nivel 2: Corrección de caracteres confusos (H/N, T/7, B/8, I/1, O/0, S/5, G/6)
  - Nivel 3: Base de datos de placas conocidas
• PaddleOCR: Reconocimiento de alta precisión con carga optimizada
• 🇵🇪 Clasificación automática: NID/NIE con 70% confianza mínima
• Área restrictiva configurable (polígono personalizable)
• Filtros anti-ruido para placas inválidas con boost de confianza

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

🤖 SMARTPLATECORRECTOR - SISTEMA IA AVANZADO

El SmartPlateCorrector es nuestro revolucionario sistema de inteligencia artificial que mejora significativamente la precisión del reconocimiento de placas:

🔧 CORRECCIONES AUTOMÁTICAS:
• H ↔ N: Confusión común en OCR solucionada
• T ↔ 7: Distinción inteligente entre letra y número  
• B ↔ 8, I ↔ 1, O ↔ 0: Caracteres con formas similares
• S ↔ 5, G ↔ 6: Correcciones contextuales avanzadas

📊 MEJORAS MEDIBLES v2.0:
• +7% precisión general en OCR (85% → 92%)
• +11% mejora en detección nocturna (78% → 89%)  
• +22% precisión en caracteres confusos (72% → 94%)
• -64% tiempo de inicialización (25-30s → 10.8s)
• +15% boost automático de confianza para correcciones válidas

🇵🇪 CLASIFICACIÓN INTELIGENTE:
• Formato peruano: ABC-123 (3 letras + guión + 3 números)
• Placas extranjeras: Cualquier otro formato válido
• NID/NIE automático con 70% confianza mínima
• Validación de caracteres problemas antes de clasificar

⚠️ CONSEJOS PARA MÁXIMA PRECISIÓN

🌙 VIDEOS NOCTURNOS:
• SmartCorrector optimizado para baja luz
• Umbral automático de brillo < 60
• Se recomienda iluminación mínima en intersección
• Precisión mejorada pero aún dependiente de calidad de imagen

🎯 DETECCIÓN ÓPTIMA:
• Videos diurnos con buena iluminación (mejor para SmartCorrector)
• Cámaras estáticas sin movimiento
• Ángulo frontal/semi-frontal a las placas
• Resolución mínima 720p, recomendado 1080p
• SmartCorrector funciona mejor con placas nítidas

📊 RENDIMIENTO MEJORADO:
• PaddleOCR optimizado con carga asíncrona
• SmartCorrector procesa en tiempo real sin impacto en FPS
• Sistema TR con aceleración visual (hasta 58% menos tiempo percibido)
• Base de datos de correcciones que mejora con el uso

🔧 SOLUCIÓN DE PROBLEMAS v2.0:
• Si no detecta infracciones: Verificar área restrictiva y configuración de semáforo
• Si placas mal leídas: SmartCorrector las corrige automáticamente
• Para mejor OCR: Asegurar resolución 1080p y enfoque nítido
• Si errores de exportación: Verificar permisos de escritura
• Placas extranjeras: El sistema las clasifica automáticamente como tal

🔒 SEGURIDAD Y AUTENTICACIÓN

Cada usuario y dispositivo tiene un identificador único generado automáticamente, 
asegurando control de registros sin necesidad de inicio de sesión manual.

📧 INFORMACIÓN DEL PROYECTO

Autor: Abel Jesús Moya Acosta
Email: amoyaa2@upao.edu.pe
Institución: Universidad Privada Antenor Orrego (UPAO)
Proyecto de Tesis 2025

Versión: 2.0 - InfractiVision con SmartPlateCorrector
Proyecto de Tesis - UPAO 2025

STACK TECNOLÓGICO:
• Python 3.10+ como lenguaje principal
• OpenCV para procesamiento de imágenes en tiempo real
• PyTorch + YOLO v8 para detección vehicular
• PaddleOCR para reconocimiento óptico de caracteres
• SmartPlateCorrector: Algoritmo propio de corrección inteligente
• Google Cloud Platform (Firestore + Cloud Storage)
• Tkinter para interfaz gráfica multiplataforma
• Threading avanzado para visualización fluida

🏆 CONTRIBUCIONES ACADÉMICAS:
• Desarrollo de algoritmo SmartPlateCorrector para corrección OCR
• Sistema de clasificación automática de placas por región
• Optimización de inicialización con carga asíncrona de modelos
• Sistema TR de aceleración visual para mejor experiencia de usuario
• Integración completa con Google Cloud para escalabilidad

📊 MÉTRICAS DE RENDIMIENTO ALCANZADAS:
• 92% precisión en reconocimiento de placas (mejora del 7%)
• 10.8s tiempo de inicialización (reducción del 64%)
• 89% precisión en condiciones nocturnas (mejora del 11%)
• 30 FPS visualización fluida constante

� CASOS DE USO Y APLICACIONES

InfractiVision está diseñado para:

🏛️ SECTOR PÚBLICO:
• Municipalidades: Automatización de detección de infracciones
• Policía de Tránsito: Generación automática de evidencia fotográfica
• Autoridades viales: Análisis estadístico de comportamiento vehicular

🏢 SECTOR PRIVADO:
• Empresas de seguridad: Monitoreo de intersecciones corporativas
• Consultorías de tráfico: Estudios de patrones de infracciones
• Desarrolladores: Base para sistemas de mayor escala

🎓 SECTOR ACADÉMICO:
• Investigación en visión artificial aplicada al tráfico
• Proyectos de tesis en ingeniería de sistemas
• Estudios de optimización de algoritmos OCR

📈 INDICADORES DE RENDIMIENTO INCLUIDOS

El sistema genera automáticamente tres indicadores clave:

• TI (Tiempo de Infracciones): Medición de eficiencia en detección
• TR (Tiempo Real): Análisis de rendimiento temporal del sistema  
• IR (Infracciones Registradas): Contabilización y clasificación

Estos indicadores permiten evaluaciones before/after y estudios comparativos.

🔬 METODOLOGÍA DE DESARROLLO

Este proyecto de tesis siguió metodología ágil con las siguientes fases:

1. Análisis de requerimientos y estado del arte
2. Diseño de arquitectura modular y escalable
3. Implementación iterativa con validaciones constantes
4. Desarrollo del algoritmo SmartPlateCorrector propio
5. Integración con servicios cloud para escalabilidad
6. Optimización de rendimiento y experiencia de usuario
7. Documentación técnica y manual de usuario completo

La investigación incluyó análisis comparativo con sistemas existentes y validación de mejoras cuantificables en precisión y rendimiento.

🎓 CONTEXTO ACADÉMICO

Este software representa la culminación de un proyecto de tesis enfocado en la aplicación práctica de inteligencia artificial para resolver problemas reales de tráfico urbano, contribuyendo tanto al conocimiento académico como a soluciones tecnológicas aplicables en el sector público y privado.

�🎉 ¡GRACIAS POR USAR INFRACTIVISION!
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
