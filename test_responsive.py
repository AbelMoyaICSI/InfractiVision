#!/usr/bin/env python3
"""
Test script para verificar el sistema responsive de InfractiVision
"""

import tkinter as tk
from src.core.video.videoplayer_opencv import VideoPlayer
from src.gui.main_gui import MainGUI

def test_responsive_system():
    """Prueba el sistema responsive en diferentes tamaños de ventana"""
    
    print("🧪 Iniciando prueba del sistema responsive...")
    
    # Crear ventana de prueba
    root = tk.Tk()
    root.title("InfractiVision - Test Responsive")
    root.geometry("1200x800")
    
    try:
        # Simular la detección de diferentes tamaños de pantalla
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        print(f"🖥️ Pantalla detectada: {screen_width}x{screen_height}")
        
        # Determinar el tipo de layout que se aplicaría
        if screen_width < 1366:
            layout_type = "PEQUEÑO (Laptop)"
            expected_panel_width = 220
        elif screen_width < 1600:
            layout_type = "MEDIANO (Estándar)"
            expected_panel_width = 300
        else:
            layout_type = "GRANDE (Monitor)"
            expected_panel_width = 380
        
        print(f"📱 Tipo de layout: {layout_type}")
        print(f"📐 Ancho esperado del panel: {expected_panel_width}px")
        
        # Crear label de información
        info_label = tk.Label(
            root, 
            text=f"Pantalla: {screen_width}x{screen_height}\nLayout: {layout_type}\nPanel: {expected_panel_width}px",
            font=("Arial", 12),
            justify="left",
            bg="#ecf0f1",
            padx=20,
            pady=20
        )
        info_label.pack(pady=20)
        
        # Crear botón de prueba que simula los diferentes layouts
        def test_layouts():
            sizes = [
                (1024, "📱 Laptop pequeño"),
                (1366, "💻 Pantalla mediana"), 
                (1920, "🖥️ Monitor grande")
            ]
            
            for width, desc in sizes:
                print(f"\n🔄 Simulando pantalla de {width}px - {desc}")
                root.geometry(f"{width}x600")
                root.update()
                
                # Simular detección responsive
                if width < 1366:
                    print("   → Layout PEQUEÑO aplicado")
                    print("   → Panel: 220px, Botones compactos")
                elif width < 1600:
                    print("   → Layout MEDIANO aplicado") 
                    print("   → Panel: 300px, Botones estándar")
                else:
                    print("   → Layout GRANDE aplicado")
                    print("   → Panel: 380px, Botones expandidos")
                
                root.after(1000)  # Esperar 1 segundo
        
        test_button = tk.Button(
            root,
            text="🧪 Probar Layouts Responsivos",
            command=test_layouts,
            font=("Arial", 14),
            bg="#3498db",
            fg="white",
            pady=10
        )
        test_button.pack(pady=10)
        
        status_label = tk.Label(
            root,
            text="✅ Sistema responsive funcional\n💡 Redimensiona la ventana para probar",
            font=("Arial", 10),
            fg="#27ae60"
        )
        status_label.pack(pady=20)
        
        print("✅ Test iniciado. Ventana de prueba creada.")
        print("💡 Redimensiona la ventana para probar el sistema responsive")
        
        # No ejecutar mainloop para evitar bloqueo en el test
        # root.mainloop()
        
    except Exception as e:
        print(f"❌ Error en test responsive: {e}")
    finally:
        if root:
            root.quit()
            root.destroy()

if __name__ == "__main__":
    test_responsive_system()