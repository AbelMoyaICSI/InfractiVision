# 🛠️ Scripts de Utilidad - InfractiVision

Esta carpeta contiene scripts auxiliares para mantenimiento, debugging y validación del sistema.

## 📋 Scripts Disponibles

### `regenerar_indicadores.py`
**Propósito**: Regenerar el archivo `indicadores_rendimiento.json` con cálculos corregidos.

**Uso**:
```bash
python scripts/regenerar_indicadores.py
```

**Funcionalidad**:
- Lee infracciones desde `data/infracciones.json` y `data/nie_infracciones.json`
- Extrae tiempos de procesamiento individuales
- Regenera los indicadores de rendimiento con cálculos actualizados
- Útil después de correcciones en la lógica de cálculo

---

### `verificar_tr_consistencia.py`
**Propósito**: Verificar que los tiempos TR (Tiempo Real) del panel coincidan con los datos en Firestore.

**Uso**:
```bash
python scripts/verificar_tr_consistencia.py
```

**Funcionalidad**:
- Compara los tiempos de procesamiento mostrados en el panel local
- Valida que coincidan con los valores que se migran a Firestore
- Muestra un resumen detallado de todas las infracciones
- Útil para debugging de sincronización de datos

---

## 📝 Notas

- Estos scripts son **herramientas de desarrollo** y no forman parte del flujo principal de la aplicación
- Se ejecutan manualmente cuando se necesita mantenimiento o validación de datos
- No son necesarios para el funcionamiento normal de InfractiVision
