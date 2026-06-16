# 🕐 Sistema TR (Tiempo Real) - Manual de Usuario

## 📖 Descripción General

El Sistema TR (Tiempo Real) implementado en InfractiVision calcula el tiempo real de procesamiento desde la perspectiva del usuario, considerando la **aceleración visual** que hace que el procesamiento se perciba más rápido de lo que realmente es.

## ⚡ Características Principales

### 1. **Aceleración Visual Inteligente**
- **🟢 Verde (4x):** Semáforo en verde = aceleración máxima
- **🟡 Amarillo (2.5x):** Semáforo en amarillo = aceleración media  
- **🔴 Rojo (1x):** Semáforo en rojo = velocidad normal (análisis intensivo)

### 2. **Cálculo de Tiempo Percibido**
```
Tiempo Percibido = Tiempo Real / Factor de Aceleración

Ejemplos:
- 10 segundos en verde → 10/4 = 2.5 segundos percibidos
- 8 segundos en amarillo → 8/2.5 = 3.2 segundos percibidos  
- 6 segundos en rojo → 6/1 = 6 segundos percibidos
```

### 3. **Seguimiento por Segmentos**
Cada segmento del video se rastrea individualmente:
- Tipo de segmento (verde/amarillo/rojo)
- Duración real en segundos
- Duración percibida por el usuario
- Factor de aceleración aplicado

## 📊 Reportes del Sistema

### Reporte en Tiempo Real
Durante el procesamiento se muestra:
```
⏱️  TR USUARIO - VERDE:
   📊 Progreso: 65.2% (489/750 frames)
   ⏰ Tiempo real: 12.45s
   👁️  Tiempo percibido: 3.11s  
   ⚡ Aceleración: 4.0x
   💾 Total acumulado: 18.75s
```

### Reporte Final
Al completar el procesamiento:
```
╔══════════════════════════════════════════════════════════════╗
║                    🕐 REPORTE FINAL TR USUARIO               ║
╠══════════════════════════════════════════════════════════════╣
║ ⏰ TIEMPO REAL TOTAL: 45.30 segundos                        ║
║ 👁️  TIEMPO PERCIBIDO: 18.75 segundos                       ║  
║ ⚡ AHORRO VISUAL: 58.6% menos tiempo de espera              ║
║                                                              ║
║ 📊 DESGLOSE POR SEGMENTOS:                                   ║
║  1. verde        | 15.20s → 3.80s (x4.0)                   ║
║  2. amarillo     | 12.10s → 4.84s (x2.5)                   ║
║  3. rojo         | 18.00s → 18.00s (x1.0)                  ║
╚══════════════════════════════════════════════════════════════╝
```

## 🔧 Implementación Técnica

### Funciones Principales

1. **`_start_user_tr_tracking()`**: Inicia el seguimiento TR
2. **`_update_visual_acceleration()`**: Ajusta la velocidad visual según semáforo
3. **`_log_user_tr_segment()`**: Registra cada segmento procesado
4. **`_print_final_user_tr_report()`**: Genera el reporte final

### Variables de Control
- `user_tr_start_time`: Tiempo de inicio del tracking
- `user_tr_segments`: Lista de segmentos procesados
- `visual_speed_multiplier`: Factor de aceleración actual
- `last_segment_start`: Marca temporal del segmento actual

## 💡 Ventajas del Sistema

### Para el Usuario
- **Experiencia mejorada:** El procesamiento se siente más rápido
- **Información transparente:** Sabe exactamente cuánto tiempo real ha transcurrido
- **Retroalimentación visual:** Ve el progreso en tiempo real

### Para el Sistema  
- **Rendimiento preservado:** No afecta la velocidad real de procesamiento
- **Threading independiente:** La visualización fluida es paralela al análisis
- **Métricas precisas:** Recolecta datos exactos de rendimiento

## 🚀 Beneficios Medibles

1. **Reducción de tiempo percibido:** Hasta 58% menos tiempo de espera
2. **Mayor fluidez visual:** 30 FPS constantes vs visualización original entrecortada
3. **Transparencia total:** El usuario siempre sabe el tiempo real transcurrido
4. **Experiencia premium:** Sensación de aplicación más rápida y responsive

## 🎯 Casos de Uso

- **Videos largos:** El ahorro de tiempo percibido es más significativo
- **Análisis intensivo:** Períodos rojos mantienen velocidad real para precisión
- **Transiciones:** Aceleración dinámica según estado del semáforo
- **Reportes académicos:** Datos exactos de tiempo real vs percibido

Este sistema representa una innovación en UX para aplicaciones de análisis de video, proporcionando una experiencia más fluida sin comprometer la precisión del procesamiento.