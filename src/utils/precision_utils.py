"""
Configuración rápida para habilitar las mejoras de precisión en InfractiVision
"""

def enable_enhanced_precision_mode(videoplayer_instance):
    """
    Habilita el modo de precisión mejorada en una instancia de VideoPlayerOpenCV
    
    Args:
        videoplayer_instance: Instancia de VideoPlayerOpenCV
    """
    if videoplayer_instance:
        # Habilitar validación de precisión
        videoplayer_instance.enable_precision_validation(True)
        
        print("🚀 MODO PRECISIÓN MEJORADA ACTIVADO")
        print("=" * 50)
        print("✅ Validación de calidad de detecciones: HABILITADA")
        print("✅ Corrección de caracteres confusos: ACTIVA")
        print("✅ Validación de formatos peruanos: ACTIVA")
        print("✅ Penalización por condiciones nocturnas: ACTIVA")
        print("=" * 50)
        print("\n📊 Para ver reporte de calidad usa:")
        print("videoplayer.get_detection_quality_report()")
        
        return True
    else:
        print("❌ Error: Instancia de videoplayer no válida")
        return False

def disable_enhanced_precision_mode(videoplayer_instance):
    """
    Deshabilita el modo de precisión mejorada (volver al modo tradicional)
    
    Args:
        videoplayer_instance: Instancia de VideoPlayerOpenCV
    """
    if videoplayer_instance:
        videoplayer_instance.enable_precision_validation(False)
        
        print("ℹ️ MODO TRADICIONAL RESTAURADO")
        print("=" * 50)
        print("🔄 TI usará conteo simple de detecciones")
        print("🔄 Sin validación de calidad adicional")
        
        return True
    else:
        print("❌ Error: Instancia de videoplayer no válida")
        return False

def show_precision_comparison(videoplayer_instance):
    """
    Muestra comparación entre modo tradicional y modo de precisión
    
    Args:
        videoplayer_instance: Instancia de VideoPlayerOpenCV
    """
    if not videoplayer_instance:
        print("❌ Error: Instancia de videoplayer no válida")
        return
        
    print(videoplayer_instance.get_detection_quality_report())
    
def quick_test_enhancements():
    """Prueba rápida de las mejoras sin necesidad de videoplayer"""
    from src.core.processing.plate_ocr_enhancer import enhance_plate_recognition
    
    print("🧪 PRUEBA RÁPIDA DE MEJORAS")
    print("=" * 40)
    
    # Casos típicos de placas con errores
    test_cases = [
        "A8C123",  # 8 vs B
        "XY5123",  # 5 vs S  
        "L0M456",  # L vs I, 0 vs O
        "AB4567",  # 4 vs A
        "T7M456",  # 7 vs T
    ]
    
    for original in test_cases:
        result = enhance_plate_recognition(None, original, False)
        enhanced = result['enhanced_text']
        confidence = result['confidence']
        
        status = "✅" if enhanced != original else "➖"
        print(f"{status} '{original}' → '{enhanced}' (conf: {confidence:.2f})")
    
    print("\n✅ Prueba completada")

if __name__ == "__main__":
    print("🔧 UTILIDADES DE PRECISIÓN MEJORADA")
    print("=" * 50)
    print("Este módulo proporciona funciones para:")
    print("1. enable_enhanced_precision_mode() - Activar modo preciso")
    print("2. disable_enhanced_precision_mode() - Volver al modo tradicional")  
    print("3. show_precision_comparison() - Ver comparación detallada")
    print("4. quick_test_enhancements() - Prueba rápida")
    print("\nEjecutando prueba rápida...")
    quick_test_enhancements()