# 📊 Estructura de Firestore - InfractiVision

## 🗂️ Colecciones Principales

### **Ruta completa en Firestore:**

```
usuarios/
  └── {user_id}/
      └── videos/
          └── {nombre_video}/
              └── configuraciones/
                  └── {config_semaforo}/
                      ├── infracciones/
                      │   └── {doc_id}
                      └── indicadores/
                          └── resumen
```

**Ejemplo real:**
```
usuarios/
  └── 31e655f-ec40-46bf-9d31-7ba89fa4e168/
      └── videos/
          └── Traffic_IP_Camera_video.mp4/
              └── configuraciones/
                  └── 10-3-15/
                      ├── infracciones/
                      │   ├── TAS-968_00-23
                      │   ├── APH-188_00-18
                      │   └── T1G-194_00-26
                      └── indicadores/
                          └── resumen
```

---

### 1. `infracciones` - Infracciones Individuales

**Ruta:** `usuarios/{user_id}/videos/{nombre_video}/configuraciones/{config_semaforo}/infracciones/{doc_id}`

Cada documento representa **UNA infracción detectada** (un vehículo en luz roja).

```json
{
  "id": "auto-generado-por-firestore",
  "placa": "TAS-968",
  "fecha": "2025-10-28",
  "hora": "19:29:10",
  "timestamp": 23.04,
  "video_timestamp": "00:23",
  "ubicacion": "Av.Grau 123",
  "avenida": "Av.Grau 123",
  "tipo_semaforo": "Semáforo en rojo",
  "franja_horaria": "07:00 - 08:00",
  
  // 📹 Información de video y configuración
  "nombre_video": "Traffic_IP_Camera_video.mp4",
  "config_semaforo": "10-3-15",
  "tiempo_verde": 10,
  "tiempo_amarillo": 3,
  "tiempo_rojo": 15,
  
  // 🎯 TI - Tasa de Identificación (confianza individual) 
  "confianza": 0.95,
  
  // Tiempo de registro
  "tiempo_procesamiento": 23.04,
  
  // Metadatos de detección
  "confidence": 0.95,
  "bbox": [1699, 0, 1919, 242],
  "track_id": 28,
  "frame": 576,
  "validation_method": "intelligent_tracking",
  "semaphore_state": "red",
  
  // Rutas de imágenes (locales)
  "plate_path": "C:\\...\\placas\\plate_TAS-968.jpg",
  "vehicle_path": "C:\\...\\autos\\vehicle_TAS-968.jpg",
  "plate_url": "https://storage.googleapis.com/.../placas/TAS-968_00-23.jpg",
  "vehicle_url": "https://storage.googleapis.com/.../vehiculos/TAS-968_00-23.jpg",
  
  // Sistema de origen
  "sistema_version": "InfractiVision_v2.0_Optimized",
  "user_id": "uuid-usuario",
  "device_id": "uuid-dispositivo",
  "username": "Abel",
  "hostname": "DESKTOP-PC",
  
  // Timestamps de creación
  "uploaded_at": "2025-10-28T19:29:10Z"
}
```

---

### 2. `indicadores` - Métricas Agregadas por Sesión

**Ruta:** `usuarios/{user_id}/videos/{nombre_video}/configuraciones/{config_semaforo}/indicadores/resumen`

Cada documento representa **métricas de una sesión de análisis** (un video procesado con una configuración específica).

```json
{
  "id": "auto-generado-por-firestore",
  "fecha": "2025-10-28",
  "hora_inicio": "19:29:10",
  "hora_fin": "19:30:25",
  
  // 📹 Información de video y configuración
  "nombre_video": "Traffic_IP_Camera_video.mp4",
  "config_semaforo": "10-3-15",
  "verde": 10,
  "amarillo": 3,
  "rojo": 15,
  "duracion_video_segundos": 30,
  "ubicacion": "Av.Grau 123",
  
  // 📊 MÉTRICAS PRINCIPALES
  "total_infracciones": 4,        // Total de vehículos detectados
  
  // 🎯 NID - Número de Infracciones Detectadas (cantidad de vehículos registrados)
  "nid": 4,                       // Mismo valor que cantidad_placas
  
  // 🎯 TI - Tasa de Identificación con desglose individual
  "ti_promedio": 0.9400,          // Promedio calculado: (0.94+0.96+0.95+0.94)/4
  "ti_porcentaje": 94.00,         // Porcentaje: 94%
  "ti_individual": [0.94, 0.96, 0.95, 0.94],  // ✅ Array con TI de cada infracción
  
  // ⏱️ TR - Tiempo de Registro con desglose individual
  "tr_total_segundos": 159,       // Suma: 22+96+19+22 = 159s
  "tr_total_minutos": 2.65,       // Total en minutos: 159/60 = 2.65min
  "tr_promedio_segundos": 39.75,  // Promedio: 159/4 = 39.75s
  "tr_individual_segundos": [22, 96, 19, 22],     // ✅ Array con TR en segundos
  "tr_individual_minutos": [0.37, 1.60, 0.32, 0.37],  // ✅ Array con TR en minutos
  
  // 🚗 Detalle de placas con desglose
  "placas_detectadas": ["A3K-861", "AV6-190", "APH-188", "T1G-194"],  // ✅ Array de placas
  "cantidad_placas": 4,
  "timestamps_video": ["00:22", "01:36", "00:19", "00:22"],  // ✅ Posiciones en video
  "calidades_deteccion": ["Excelente", "Excelente", "Excelente", "Excelente"],  // ✅ Calidad de cada detección
  
  // 📈 Métricas de calidad agregadas
  "detecciones_excelentes": 4,    // TI >= 95%
  "detecciones_buenas": 0,        // 85% <= TI < 95%
  "detecciones_aceptables": 0,    // 70% <= TI < 85%
  "detecciones_bajas": 0,         // TI < 70%
  
  // Sistema
  "sistema_version": "InfractiVision_v2.0_Optimized",
  "procesado_por": "PreprocesadorInteligente",
  "user_id": "uuid-usuario",
  "device_id": "uuid-dispositivo",
  "username": "Abel",
  "hostname": "DESKTOP-PC",
  
  // Timestamps
  "fecha_subida": "20251028T193025Z",
  "storage_url": "https://storage.googleapis.com/...",
  "created_at": "2025-10-28T19:30:25Z",
  "updated_at": "2025-10-28T19:30:25Z"
}
```

---

## 🔄 Ventajas de la estructura por Video y Configuración

✅ **Organización jerárquica**: Cada video tiene su propia rama  
✅ **Múltiples configuraciones**: Puedes analizar el mismo video con diferentes tiempos de semáforo  
✅ **Sin sobrescritura**: Cada análisis queda guardado por separado  
✅ **Historial completo**: Puedes consultar análisis anteriores sin perder datos  
✅ **Escalable**: Soporta múltiples videos y configuraciones sin conflictos  
✅ **Compatible con backend**: La misma estructura que usa `infracti_backend.py`

---

## 🔄 Migración de Datos

### Mapeo de campos antiguos → nuevos

| Campo Antiguo | Campo Nuevo | Notas |
|--------------|-------------|-------|
| `clasificacion: "NID"` | ❌ Eliminado | Ahora TODAS las detecciones son válidas |
| `clasificacion: "NIE"` | ❌ Eliminado | Ya no hay detecciones "no válidas" |
| `confidence` | `ti` | Mismo valor, renombrado para claridad |
| `nid_count` | `total_infracciones` | Cuenta total de registros |
| `nid_porcentaje` | `ti_porcentaje` | Porcentaje de confianza promedio |
| `nie_count` | ❌ Eliminado | Ya no existe |

---

## 📝 Ejemplo Completo de una Sesión

### Infracciones (4 documentos)

```json
// Documento 1
{
  "placa": "TAS-968",
  "ti": 0.95,
  "ti_porcentaje": 95,
  "calidad_deteccion": "Excelente",
  "tr_minutos": 0.384,
  // ... otros campos
}

// Documento 2
{
  "placa": "APH-188",
  "ti": 0.97,
  "ti_porcentaje": 97,
  "calidad_deteccion": "Excelente",
  "tr_minutos": 0.315,
  // ... otros campos
}

// Documento 3
{
  "placa": "T1G-194",
  "ti": 0.99,
  "ti_porcentaje": 99,
  "calidad_deteccion": "Excelente",
  "tr_minutos": 0.437,
  // ... otros campos
}

// Documento 4
{
  "placa": "A3K-961",
  "ti": 0.95,
  "ti_porcentaje": 95,
  "calidad_deteccion": "Excelente",
  "tr_minutos": 0.368,
  // ... otros campos
}
```

### Indicadores (1 documento agregado)

```json
{
  "total_infracciones": 4,
  
  // NID - Número de Infracciones Detectadas
  "nid": 4,
  
  // TI con desglose individual
  "ti_individual": [0.95, 0.97, 0.99, 0.95],
  "ti_promedio": 0.9650,        // (0.95+0.97+0.99+0.95)/4 = 0.965
  "ti_porcentaje": 96.50,
  
  // TR con desglose individual en segundos y minutos
  "tr_individual_segundos": [23, 19, 26, 22],
  "tr_individual_minutos": [0.38, 0.32, 0.43, 0.37],
  "tr_total_segundos": 90,      // 23+19+26+22
  "tr_total_minutos": 1.50,     // 90/60 = 1.5
  "tr_promedio_segundos": 22.50,// 90/4 = 22.5
  
  // Desglose de placas y calidades
  "placas_detectadas": ["TAS-968", "APH-188", "T1G-194", "A3K-961"],
  "cantidad_placas": 4,
  "timestamps_video": ["00:23", "00:19", "00:26", "00:22"],
  "calidades_deteccion": ["Excelente", "Excelente", "Excelente", "Excelente"],
  
  // Contadores de calidad
  "detecciones_excelentes": 4,
  "detecciones_buenas": 0,
  "detecciones_aceptables": 0,
  "detecciones_bajas": 0,
  
  // Sistema
  "user_id": "uuid-usuario",
  "device_id": "uuid-dispositivo",
  "username": "Abel",
  "hostname": "DESKTOP-PC",
  "fecha_subida": "20251028T193025Z",
  "storage_url": "https://storage.googleapis.com/..."
}
```

---

## 🎯 Ventajas de la Nueva Estructura

✅ **Organización jerárquica por video**: Cada video analizado tiene su propia rama  
✅ **Múltiples configuraciones**: Analiza el mismo video con diferentes tiempos de semáforo  
✅ **Desglose individual completo**: Arrays `ti_individual`, `tr_individual_segundos`, `tr_individual_minutos`  
✅ **Sin sobrescritura de datos**: Cada análisis queda guardado permanentemente  
✅ **Consulta única eficiente**: Toda la info de una sesión en un solo documento  
✅ **Historial completo**: Puedes comparar análisis del mismo video con diferentes configs  
✅ **Análisis estadístico avanzado**: Media, mediana, desviación estándar sobre arrays  
✅ **Trazabilidad completa**: Relación placa ↔ TI ↔ TR ↔ timestamp ↔ video ↔ config preservada  
✅ **Compatible con backend**: Usa la misma estructura que `infracti_backend.py`  
✅ **Escalable**: Soporta múltiples usuarios, videos y configuraciones sin conflictos  

---

## 📊 Consultas Firestore Útiles

### Obtener todas las infracciones de un video específico
```javascript
db.collection('usuarios')
  .doc(userId)
  .collection('videos')
  .doc('Traffic_IP_Camera_video.mp4')
  .collection('configuraciones')
  .doc('10-3-15')
  .collection('infracciones')
  .get()
```

### Obtener indicadores de un video y configuración
```javascript
db.collection('usuarios')
  .doc(userId)
  .collection('videos')
  .doc('Traffic_IP_Camera_video.mp4')
  .collection('configuraciones')
  .doc('10-3-15')
  .collection('indicadores')
  .doc('resumen')
  .get()
  .then(doc => {
    const data = doc.data();
    
    // Acceder a arrays desglosables
    console.log('TI individual:', data.ti_individual);
    console.log('TR individual (s):', data.tr_individual_segundos);
    console.log('TR individual (min):', data.tr_individual_minutos);
    console.log('Placas:', data.placas_detectadas);
    console.log('Calidades:', data.calidades_deteccion);
    
    // Reconstruir tabla desglosable
    for (let i = 0; i < data.placas_detectadas.length; i++) {
      console.log({
        placa: data.placas_detectadas[i],
        ti: data.ti_individual[i],
        tr_seg: data.tr_individual_segundos[i],
        tr_min: data.tr_individual_minutos[i],
        timestamp: data.timestamps_video[i],
        calidad: data.calidades_deteccion[i]
      });
    }
  })
```

### Listar todos los videos analizados por un usuario
```javascript
db.collection('usuarios')
  .doc(userId)
  .collection('videos')
  .get()
  .then(snapshot => {
    snapshot.forEach(doc => {
      console.log('Video:', doc.id);
    });
  })
```

### Listar todas las configuraciones de un video
```javascript
db.collection('usuarios')
  .doc(userId)
  .collection('videos')
  .doc('Traffic_IP_Camera_video.mp4')
  .collection('configuraciones')
  .get()
  .then(snapshot => {
    snapshot.forEach(doc => {
      console.log('Configuración:', doc.id);  // Ej: "10-3-15"
    });
  })
```

### Obtener infracciones con TI > 95% de un video
```javascript
db.collection('usuarios')
  .doc(userId)
  .collection('videos')
  .doc('Traffic_IP_Camera_video.mp4')
  .collection('configuraciones')
  .doc('10-3-15')
  .collection('infracciones')
  .where('confianza', '>', 0.95)
  .orderBy('confianza', 'desc')
  .get()
```

### Comparar resultados de diferentes configuraciones del mismo video
```javascript
// Obtener indicadores de configuración 1
const config1 = await db.collection('usuarios')
  .doc(userId)
  .collection('videos')
  .doc('Traffic_IP_Camera_video.mp4')
  .collection('configuraciones')
  .doc('10-3-15')
  .collection('indicadores')
  .doc('resumen')
  .get();

// Obtener indicadores de configuración 2
const config2 = await db.collection('usuarios')
  .doc(userId)
  .collection('videos')
  .doc('Traffic_IP_Camera_video.mp4')
  .collection('configuraciones')
  .doc('30-5-40')
  .collection('indicadores')
  .doc('resumen')
  .get();

// Comparar
console.log('Config 10-3-15:', config1.data().ti_promedio);
console.log('Config 30-5-40:', config2.data().ti_promedio);
```
