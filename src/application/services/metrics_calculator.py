"""Calculadora de métricas de tesis (TI, TR, NID, NIE).

Extraído de preprocessing_dialog.py sin cambios algorítmicos.
"""

class ThesisMetricsCalculator:
    """
    Calculadora de métricas para la tesis: TI, TR, NID, NIE.
    """
    
    def __init__(self):
        self.start_time = None
        self.processing_times = []
        
    def calculate_metrics(self, infractions_data):
        """Calcula métricas completas para la tesis."""
        if not infractions_data:
            return self._empty_metrics()
            
        total_events = len(infractions_data)
        nid_events = [inf for inf in infractions_data if inf.get('clasificacion') == 'NID']
        nie_events = [inf for inf in infractions_data if inf.get('clasificacion') == 'NIE']
        
        # Calcular métricas
        nid_count = len(nid_events)
        nie_count = len(nie_events)
        nid_percentage = (nid_count / total_events * 100) if total_events > 0 else 0
        nie_percentage = (nie_count / total_events * 100) if total_events > 0 else 0
        
        # TI: Tasa de Infracciones (asumiendo que NID son válidas)
        ti_rate = nid_percentage  # Solo NID cuentan como infracciones válidas
        
        # TR: Tiempo de Registro promedio
        processing_times = [inf.get('tiempo_procesamiento', 0) for inf in infractions_data if inf.get('tiempo_procesamiento')]
        tr_average = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'TI': {
                'tasa_infracciones_validas': round(ti_rate, 2),
                'infracciones_detectadas': nid_count,
                'total_eventos': total_events
            },
            'TR': {
                'tiempo_promedio_segundos': round(tr_average, 2),
                'tiempo_promedio_minutos': round(tr_average / 60, 2),
                'muestras': len(processing_times)
            },
            'NID': {
                'cantidad': nid_count,
                'porcentaje': round(nid_percentage, 2),
                'objetivo_cumplido': nid_percentage >= 70  # Objetivo: >70% NID
            },
            'NIE': {
                'cantidad': nie_count,
                'porcentaje': round(nie_percentage, 2),
                'controlado': nie_percentage <= 30  # Objetivo: <30% NIE
            },
            'resumen_tesis': {
                'sistema_efectivo': nid_percentage >= 70 and nie_percentage <= 30,
                'confiabilidad_general': 'Alta' if nid_percentage >= 85 else 'Media' if nid_percentage >= 70 else 'Baja',
                'justificacion_nie': f"NIE controlado al {round(nie_percentage, 1)}% - Transparente vs errores humanos ocultos"
            }
        }
        
    def _empty_metrics(self):
        """Métricas vacías para casos sin datos."""
        return {
            'TI': {'tasa_infracciones_validas': 0, 'infracciones_detectadas': 0, 'total_eventos': 0},
            'TR': {'tiempo_promedio_segundos': 0, 'tiempo_promedio_minutos': 0, 'muestras': 0},
            'NID': {'cantidad': 0, 'porcentaje': 0, 'objetivo_cumplido': False},
            'NIE': {'cantidad': 0, 'porcentaje': 0, 'controlado': True},
            'resumen_tesis': {'sistema_efectivo': False, 'confiabilidad': 'Sin datos', 'justificacion_nie': 'No hay datos suficientes'}
        }
