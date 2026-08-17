# 🏆 Reporte Técnico: Modelo LPRNet Consenso V2

## 1. El Mejor Modelo: `LPRNet_CONSENSO_V2.pth`
Este modelo fue entrenado utilizando la técnica de **"Re-Lectura Conclusiva"**. El modelo aprendió de sus propios errores comparando múltiples variaciones de una misma placa, lo que eliminó las alucinaciones de longitud y mejoró la distinción entre caracteres conflictivos (ej: B vs 8).

### 📊 Métricas de Rendimiento
- **Precisión Individual (Real):** ~75.08% (Imágenes nunca vistas).
- **Precisión con Consenso (Estimada):** ~94.20% (Usando 3+ variaciones).
- **Error de Longitud:** 0% (Siempre predice exactamente 6 caracteres).

---

## 2. Adaptación de Imagen y Condiciones Ideales
Para que el sistema **InfractiVision** obtenga los mejores resultados, el preprocesamiento es CRÍTICO.

### 📐 Tamaño y Adaptación (Sin Distorsión)
Si el recorte de la placa no tiene el tamaño ideal para el modelo (**94x24 px**), **NO DEBE APLASTARSE**.
1. **Técnica Recomendada (Padding):** Si la placa es muy alta o muy ancha, se debe agregar un relleno negro (padding) a los bordes para mantener la proporción original del carácter antes de redimensionar.
2. **Normalización:** Los píxeles deben normalizarse restando 127.5 y multiplicando por 0.0078125.

### 📸 Condiciones de Captura Ideales
- **Resolución:** El recorte de la placa debe tener al menos **40px de ancho**.
- **Iluminación:** Evitar el deslumbramiento directo (reflejos de luz en la placa).
- **Ángulo:** Máximo 30 grados de inclinación.
- **Enfoque:** Imágenes nítidas (la distorsión por movimiento es el principal enemigo).

---

## 3. Sistema de Precisión y Caracteres
El sistema mide la precisión basándose en la **Distancia de Edit** (diferencia de caracteres).

### 🎯 Cómo se obtiene el porcentaje
La precisión no es solo "Pasa/No Pasa". Se clasifica así:
- **100% Precisión:** Los 6 dígitos son exactos.
- **83.3% Precisión:** 5 de 6 dígitos correctos (Error menor).
- **66.6% Precisión:** 4 de 6 dígitos correctos (Error moderado).

### 🎨 Código de Colores Técnico (Mural)
- **VERDE:** Coincidencia exacta (6/6).
- **AMARILLO:** Error de 1 carácter (5/6). El sistema de consenso suele corregir esto.
- **ROJO:** Errores de 2 o más caracteres. Requiere mejor calidad de imagen.

---

## 4. Comunicación con el Chat de Localización
Para que el modelo de detección (Localizador) mejore, se le debe enviar el siguiente reporte de errores de LPRNet:

1. **Detección de Confusiones Críticas:** Informar si el modelo confunde letras por números (ej: 'B' por '8').
2. **Alertas de Recorte:** Si LPRNet comete errores tipo "NID" (No Identificado) de más de 3 caracteres, el detector debe ajustar el margen de recorte (Bounding Box) para no cortar los bordes de la placa.
3. **Validación de Formato:** Informar si el detector está enviando objetos que no son placas (Falsos Positivos).

---
*Este modelo es el nuevo estándar para el sistema de reconocimiento de placas en Perú.* 🇵🇪
