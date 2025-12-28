
import sys
import os

# Mocking or importing the necessary logic
from collections import Counter

def super_consensus_mock(plate_texts, regional_context="Trujillo"):
    if len(plate_texts) < 2:
        return {'has_consensus': False, 'reason': 'insuficientes_frames'}
        
    normalized_plates = [p.replace('-', '').replace(' ', '').upper() for p in plate_texts]
    len_counts = Counter(len(p) for p in normalized_plates)
    target_len = len_counts.most_common(1)[0][0]
    
    valid_normalized = [p for p in normalized_plates if len(p) == target_len]
    if not valid_normalized:
        return {'has_consensus': False, 'reason': 'variación_longitud_excesiva'}
        
    final_chars = []
    for i in range(target_len):
        chars_at_pos = [p[i] for p in valid_normalized]
        char_votes = Counter(chars_at_pos)
        
        # Regional Intelligence: Prioritize 'T' in first position
        if i == 0 and 'T' in char_votes and regional_context == "Trujillo":
            best_char = 'T'
        else:
            best_char = char_votes.most_common(1)[0][0]
            
        final_chars.append(best_char)
        
    best_text = "".join(final_chars)
    if len(best_text) == 6:
        formatted_text = f"{best_text[:3]}-{best_text[3:]}"
    else:
        formatted_text = best_text
        
    return {
        'has_consensus': True,
        'best_text': formatted_text
    }

def test_consensus():
    # Test case 1: Common confusion '8' vs 'B' and '0' vs 'O'
    noisy_plates = ["T3J-53B", "T3J-538", "T3J-538", "T3I-538"]
    result = super_consensus_mock(noisy_plates)
    print(f"Test 1 (Noisy 8/B): {noisy_plates} -> {result['best_text']}")
    assert result['best_text'] == "T3J-538"
    
    # Test case 2: Regional 'T' priority
    # Suppose OCR is split between 'I' and 'T' in first pos
    regional_noisy = ["I4A-376", "T4A-376"]
    result = super_consensus_mock(regional_noisy, regional_context="Trujillo")
    print(f"Test 2 (Regional T): {regional_noisy} -> {result['best_text']}")
    assert result['best_text'].startswith("T")

    # Test case 3: 123-ABC format (inverse)
    inverse_plates = ["123-ABC", "123-A8C", "123-ABC"]
    result = super_consensus_mock(inverse_plates)
    print(f"Test 3 (Inverse): {inverse_plates} -> {result['best_text']}")
    assert result['best_text'] == "123-ABC"

    print("\n✅ ALL CONSENSUS TESTS PASSED!")

if __name__ == "__main__":
    test_consensus()
