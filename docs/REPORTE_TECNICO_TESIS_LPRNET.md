# REPORTE TÉCNICO: SUSTENTO ARQUITECTURAL LPRNET-PERÚ (SIIV 2026)

## 1. MÉTODOLOGÍA DE EVALUACIÓN CIENTÍFICA

Para la validación del sistema **InfractiVision**, se adoptan los estándares de métricas de reconocimiento de caracteres de grado industrial:

### 1.1. Exact Match (EM)
Se establece la precisión sobre la **Placa Completa**. Una detección se considera correcta (NID - Válida) únicamente si los 6 caracteres de la placa peruana coinciden en su totalidad con la realidad física.
*   **Fórmula:** $EM = \frac{\text{Detecciones Exactas}}{\text{Total de Muestras}}$

### 1.2. Character Error Rate (CER)
Métrica utilizada para cuantificar la distancia entre la predicción del modelo y la placa real, permitiendo identificar variaciones mínimas por carácter.
*   **Fórmula:** $CER = \frac{\text{Distancia Levenshtein(Predicción, Real)}}{\text{Longitud Real (6)}}$
*   **Interpretación:** Un error en un carácter representa un CER del 16.6% (Exactitud del 83.3%).

## 2. ANÁLISIS DE INCERTIDUMBRE Y FALLOS

### 2.1. Umbrales de Confianza (Thresholds)
El sistema clasifica la fiabilidad de la lectura basándose en el promedio de probabilidad del tensor de salida:
*   **Grado Industrial (> 85%):** Lectura de alta fiabilidad, no requiere intervención humana.
*   **Grado de Investigación (70% - 85%):** Requiere validación por el sistema de consenso multicanal.
*   **NIE (No Identificado):** Activado cuando la longitud es < 6 o la confianza promedio es crítica (< 70%).

### 2.2. Justificación del Sliding Window y Resolución
El modelo opera mediante una **Ventana Deslizante (Sliding Window)** sobre el mapa de características de 94x24 píxeles. 
*   **Impacto del Centrado:** Una placa descentrada causa que la ventana procese fracciones de caracteres (ej. mitad de una 'T' y mitad de una 'I'), resultando en alucinaciones del OCR. 
*   **Efecto de Compresión:** La resolución de 94x24 impone un límite físico para caracteres anchos (M, W, Q). El sistema InfractiVision mitiga esto mediante el **Protocolo de Aire Lateral (4px)** para mantener la morfología natural del carácter.

## 3. MATRIZ DE CONFUSIÓN TÍPICA (SIIV PERÚ)
Se identifican y corrigen mediante algoritmos de post-procesamiento las siguientes ambigüedades:
*   **Letras vs Números:** B ↔ 8, O ↔ 0, D ↔ 0, Z ↔ 2.
*   **Bordes de ROI:** T ↔ 7, I ↔ 1, M ↔ B.

---
*Este documento constituye el respaldo técnico para la defensa de la tesis de ingeniería sobre el Sistema InfractiVision.*
