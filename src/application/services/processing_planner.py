"""Planificador inteligente de procesamiento por segmentos del semáforo.

Extraído de preprocessing_dialog.py sin cambios algorítmicos.
"""

class IntelligentTrafficOptimizer:
    """
    Sistema de optimización inteligente basado en ciclos de semáforo.
    
    CONCEPTOS CLAVE:
    - Pre-alerta: Cuando entra AMARILLO, predice t₀ (inicio de ROJO)
    - Ventana de foco: [t₀-Δpre, t₀+Δpost] donde concentrar recursos
    - Fast-scan: Durante VERDE y primera mitad de AMARILLO (frame-skip x2/x3)
    - Full precision: Cerca de t₀ y en ROJO (detección completa + tracking + OCR)
    - Validación de perspectiva: Historial de posición para evitar falsos positivos
    """
    
    def __init__(self, cycle_durations, fps, total_frames):
        """
        Inicializa el optimizador.
        
        Args:
            cycle_durations: Dict con duración de cada fase {'green': X, 'yellow': Y, 'red': Z}
            fps: Frames per second del video
            total_frames: Total de frames del video
        """
        self.cycle_durations = cycle_durations
        self.fps = fps
        self.total_frames = total_frames
        
        # Configuración de ventanas de foco
        self.window_pre_ms = 1200   # 1.2s antes de t₀
        self.window_post_ms = 1800  # 1.8s después de t₀
        self.fast_skip_rate = 2     # Skip x2 durante fast-scan (amarillo)
        self.green_skip_rate = 3    # Skip x3 durante fase VERDE (más evidente)
        self.min_conf_vehicle = 0.45
        self.min_conf_ocr = 0.60
        
        # Cálculos base - VALIDACIÓN DEFENSIVA
        self.frames_per_state = {}
        for state, duration in cycle_durations.items():
            try:
                # Manejar diferentes formatos de duración
                if isinstance(duration, (list, tuple)):
                    # Si es una lista/tupla, tomar el primer elemento
                    duration_value = float(duration[0]) if len(duration) > 0 else 10.0
                elif isinstance(duration, (int, float)):
                    # Si es un número, usarlo directamente
                    duration_value = float(duration)
                else:
                    # Si es string u otro tipo, intentar convertir
                    duration_value = float(duration)
                
                self.frames_per_state[state] = int(duration_value * fps)
                
            except (ValueError, TypeError, IndexError) as e:
                print(f"⚠️  Error procesando duración para {state}: {duration} - usando valor por defecto")
                # Valores por defecto si hay error
                default_durations = {'green': 12, 'yellow': 2, 'red': 10}
                self.frames_per_state[state] = int(default_durations.get(state, 10) * fps)
        
        self.cycle_frames = sum(self.frames_per_state.values())
        
        # Generar plan de procesamiento
        self.processing_plan = self._generate_processing_plan()
        
    def _generate_processing_plan(self):
        """
        Genera el plan completo de procesamiento optimizado.
        
        Returns:
            List[Dict]: Plan con información de cada segmento
        """
        plan = []
        frame_index = 0
        cycle_number = 0
        
        print(f"🚀 OPTIMIZADOR INTELIGENTE: Generando plan para {self.total_frames} frames")
        print(f"   📊 Ciclo semáforo: Verde={self.frames_per_state['green']} | Amarillo={self.frames_per_state['yellow']} | Rojo={self.frames_per_state['red']}")
        
        # FALLBACK PARA VIDEOS CORTOS: Si el video es más corto que un ciclo completo
        cycle_duration = sum(self.frames_per_state.values())
        if self.total_frames < cycle_duration:
            print(f"⚠️  VIDEO CORTO DETECTADO: {self.total_frames} frames < {cycle_duration} frames del ciclo")
            print(f"🔄 APLICANDO MODO COMPATIBILIDAD: Procesamiento tradicional")
            
            # Para videos cortos, crear un segmento que cubra todo el video
            # y asumiremos que contiene al menos una fase ROJA
            plan.append({
                'type': 'focus_window',
                'phase': 'short_video_fallback',
                'start_frame': 0,
                'end_frame': self.total_frames,
                'skip_rate': 1,  # Sin skip para videos cortos
                'processing_intensity': 'maximum',
                'cycle': 0,
                't0_frame': self.total_frames // 2,  # Asumir que hay rojo en la mitad
                'is_infraction_zone': True
            })
            
            print(f"   🎯 MODO COMPATIBILIDAD: 1 segmento completo ({self.total_frames} frames)")
            return plan
        
        while frame_index < self.total_frames:
            # Calcular frames para cada fase del ciclo actual
            green_start = frame_index
            green_end = min(green_start + self.frames_per_state["green"], self.total_frames)
            
            yellow_start = green_end
            yellow_end = min(yellow_start + self.frames_per_state["yellow"], self.total_frames)
            
            red_start = yellow_end
            red_end = min(red_start + self.frames_per_state["red"], self.total_frames)
            
            # t₀ = inicio de ROJO
            t0_frame = red_start
            
            # Calcular ventana de foco alrededor de t₀
            window_pre_frames = int((self.window_pre_ms / 1000.0) * self.fps)
            window_post_frames = int((self.window_post_ms / 1000.0) * self.fps)
            
            focus_window_start = max(0, t0_frame - window_pre_frames)
            focus_window_end = min(self.total_frames, t0_frame + window_post_frames)
            
            # FASE VERDE: Fast-scan (x3 para hacer más evidente la aceleración)
            if green_start < green_end:
                plan.append({
                    'type': 'fast_scan',
                    'phase': 'green',
                    'start_frame': green_start,
                    'end_frame': green_end,
                    'skip_rate': self.green_skip_rate,  # x3 para fase verde
                    'processing_intensity': 'light',
                    'cycle': cycle_number
                })
            
            # FASE AMARILLO: Dividir en dos partes
            if yellow_start < yellow_end:
                yellow_mid = yellow_start + (yellow_end - yellow_start) // 2
                
                # Primera mitad de amarillo: Fast-scan
                plan.append({
                    'type': 'fast_scan',
                    'phase': 'yellow_early',
                    'start_frame': yellow_start,
                    'end_frame': yellow_mid,
                    'skip_rate': self.fast_skip_rate,
                    'processing_intensity': 'light',
                    'cycle': cycle_number
                })
                
                # Segunda mitad de amarillo: Pre-alerta (preparar para foco)
                plan.append({
                    'type': 'pre_alert',
                    'phase': 'yellow_late',
                    'start_frame': yellow_mid,
                    'end_frame': yellow_end,
                    'skip_rate': 1,  # Sin skip, preparando para precisión
                    'processing_intensity': 'medium',
                    'cycle': cycle_number,
                    't0_prediction': t0_frame
                })
            
            # FASE ROJO: Full precision dentro de ventana de foco
            if red_start < red_end:
                # Segmento de foco completo (incluye parte de amarillo + todo rojo)
                plan.append({
                    'type': 'focus_window',
                    'phase': 'red',
                    'start_frame': focus_window_start,
                    'end_frame': focus_window_end,
                    'skip_rate': 1,  # Sin skip
                    'processing_intensity': 'maximum',
                    'cycle': cycle_number,
                    't0_frame': t0_frame,
                    'is_infraction_zone': True
                })
            
            # Avanzar al siguiente ciclo
            frame_index = red_end
            cycle_number += 1
            
            # Prevenir loops infinitos
            if frame_index >= self.total_frames:
                break
                
        total_fast_frames = sum(p['end_frame'] - p['start_frame'] for p in plan if p['type'] == 'fast_scan')
        total_focus_frames = sum(p['end_frame'] - p['start_frame'] for p in plan if p['type'] == 'focus_window')
        
        print(f"   ⚡ OPTIMIZACIÓN: Fast-scan={total_fast_frames} frames | Full-precision={total_focus_frames} frames")
        print(f"   🎯 EFICIENCIA: {((self.total_frames - total_focus_frames) / self.total_frames) * 100:.1f}% de frames en modo rápido")
        
        return plan
    
    def get_processing_segments(self):
        """
        Retorna TODOS los segmentos en orden para un procesamiento secuencial fluido.
        
        Returns:
            List[Tuple]: Lista de (start_frame, end_frame, skip_rate, phase)
        """
        return [
            (s['start_frame'], s['end_frame'], s['skip_rate'], s['phase'])
            for s in self.processing_plan
        ]
    
    def get_segment_config(self, frame_index):
        """
        Obtiene la configuración de procesamiento para un frame específico.
        
        Args:
            frame_index: Índice del frame
            
        Returns:
            Dict: Configuración de procesamiento para ese frame
        """
        for segment in self.processing_plan:
            if segment['start_frame'] <= frame_index < segment['end_frame']:
                return segment
                
        # Fallback: configuración por defecto
        return {
            'type': 'default',
            'phase': 'unknown',
            'processing_intensity': 'medium',
            'skip_rate': 1
        }
