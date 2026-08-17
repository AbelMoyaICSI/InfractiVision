import torch
import torch.nn.functional as F
import numpy as np

def decode_fixed_length(logits, chars, length=6):
    """
    Decodificador de Longitud Fija para LPRNet (Específico para Perú SIIV).
    En lugar de usar Greedy CTC (que puede devolver menos de 6 caracteres),
    esta función busca los 6 picos de mayor probabilidad ignorando el carácter 'blank'.
    
    Args:
        logits (torch.Tensor): Salida bruta del modelo (Batch, Classes, Time)
        chars (list): Diccionario de caracteres
        length (int): Longitud esperada (6 para placas peruanas)
        
    Returns:
        str: Cadena de longitud fija
        float: Confianza ponderada
    """
    # 1. Aplicar Softmax para obtener probabilidades (Classes, Time)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    num_classes = probs.shape[0]
    num_time_steps = probs.shape[1]
    blank_idx = num_classes - 1  # Por convención, el último es el blank '-'
    
    # 2. Para cada paso de tiempo, encontrar el mejor carácter (que no sea blank)
    # y su probabilidad relativa.
    candidates = []
    for t in range(num_time_steps):
        # Probabilidad del mejor carácter que NO sea el blank
        char_probs = probs[:blank_idx, t]
        best_char_idx = np.argmax(char_probs)
        best_prob = char_probs[best_char_idx]
        
        # Guardamos: (Probabilidad, Índice del carácter, Paso de tiempo)
        candidates.append({
            'prob': best_prob,
            'char_idx': best_char_idx,
            'time': t
        })
    
    # 3. Ordenar candidatos por probabilidad de mayor a menor
    candidates.sort(key=lambda x: x['prob'], reverse=True)
    
    # 4. Tomar los 'length' mejores candidatos (los picos más fuertes)
    top_candidates = candidates[:length]
    
    # 5. Volver a ordenarlos por tiempo para mantener el orden de lectura
    top_candidates.sort(key=lambda x: x['time'])
    
    # 6. Construir el resultado y calcular la confianza media
    res_chars = [chars[c['char_idx']] for c in top_candidates]
    avg_conf = np.mean([c['prob'] for c in top_candidates])
    
    return "".join(res_chars), float(avg_conf)

