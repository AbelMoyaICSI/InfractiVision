#!/usr/bin/env python
"""
Compara resultados OCR (sin sharpen vs con sharpen) con ground truth.
"""

import json
import re
from pathlib import Path

# Ground truth cars per video (from verdad.test.json)
GT = {
    "VID2EDIT": {
        "cars": [
            "T5M212", "TEQ850", "T2Q914", "ABJ746", "T4C222", "P3S454",
            "BVJ513", "T7C065", "T9S885", "T4E110", "T5T045", "T7Q367",
            "T5X243", "T6C533"
        ]
    },
    "VID4EDIT": {
        "cars": [
            "CEK390", "T3G464", "T4Z499", "M5M516", "T5U677", "M3P299",
            "V7T222", "T70644", "T6C470", "T1T646"
        ]
    },
    "VID7EDIT": {
        "cars": [
            "F5P035", "T1D547", "T6D461", "T2V265", "T6P400", "T6E290",
            "T7U596", "T7U009", "T3H169", "T6Y544", "T7R868", "EUH662"
        ]
    }
}

# OCR results (from the runs)
OCR_NO_SHARPEN = {
    "VID2EDIT": {
        "v149_best": "BJ513",
        "v162_best": "T7C-065",
        "v171_best": "T9S-885",
        "v217_best": "T4E-110",
        "v21_best": "T5M-212",
        "v220_best": "T6C5338",
        "v227_best": "T7Q-367",
        "v228_best": "T5X-243",
        "v29_best": "A8J-746",
        "v32_best": "T7F-234",
        "v51_best": "T4C22",
        "v83_best": "B3S4546",
    },
    "VID4EDIT": {
        "v1_best": "T5U-677",
        "v232_best": "T70-644",
        "v238_best": "T6C70",
        "v63_best": "CEK3907",
        "v68_best": "T7",
        "v69_best": "AS4",
        "v73_best": "T4Z-499",
        "v78_best": "V7T-222",
        "v80_best": "T3P-299",
    },
    "VID7EDIT": {
        "v14_best": "T6P00",
        "v19_best": "T4S-298",
        "v1_best": "TT46",
        "v21_best": "M1A-636",
        "v23_best": "T6E-298",
        "v24_best": "T7U-596",
        "v28_best": "T7U-009",
        "v35_best": "T6Y-544",
        "v9_best": "T83",
    }
}

OCR_WITH_SHARPEN = {
    "VID2EDIT": {
        "v149_best": "B513",
        "v162_best": "T7C0658",
        "v171_best": "T9S-885",
        "v217_best": "T4E-110",
        "v21_best": "T5M-212",
        "v220_best": "TC5-338",
        "v227_best": "T7Q-367",
        "v228_best": "T5X-243",
        "v29_best": "AEJ-746",
        "v32_best": "T7F-234",
        "v51_best": "T4C-222",
        "v83_best": "A3S4549",
    },
    "VID4EDIT": {
        "v1_best": "T5U67",
        "v232_best": "T70-644",
        "v238_best": "T6C-470",
        "v63_best": "CEK-327",
        "v68_best": "T7",
        "v69_best": "CA6",
        "v73_best": "T4499",
        "v78_best": "TT222",
        "v80_best": "T3P-299",
    },
    "VID7EDIT": {
        "v14_best": "T6009",
        "v19_best": "T4S-298",
        "v1_best": "TS446",
        "v21_best": "M1A-636",
        "v23_best": "T6E2792",
        "v24_best": "T7U-596",
        "v28_best": "T7U-009",
        "v35_best": "T6Y-544",
        "v9_best": "TC3",
    }
}

# File → GT plate mapping (approximate by order)
FILE_TO_GT_PLATE = {
    "VID2EDIT": {
        "v149_best": "BVJ513",
        "v162_best": "T7C065",
        "v171_best": "T9S885",
        "v217_best": "T4E110",
        "v21_best": "T5M212",
        "v220_best": "T6C533",
        "v227_best": "T7Q367",
        "v228_best": "T5X243",
        "v29_best": "ABJ746",
        "v32_best": "T5T045",
        "v51_best": "T4C222",
        "v83_best": "P3S454",
    },
    "VID4EDIT": {
        "v1_best": "T5U677",
        "v232_best": "T70644",
        "v238_best": "T6C470",
        "v63_best": "CEK390",
        "v68_best": "T1T646",
        "v69_best": "M3P299",
        "v73_best": "T4Z499",
        "v78_best": "V7T222",
        "v80_best": "M5M516",
    },
    "VID7EDIT": {
        "v14_best": "T6P400",
        "v19_best": "T3H169",
        "v1_best": "F5P035",
        "v21_best": "EUH662",
        "v23_best": "T6E290",
        "v24_best": "T7U596",
        "v28_best": "T7U009",
        "v35_best": "T6Y544",
        "v9_best": "T7R868",
    }
}


def normalize(plate: str) -> str:
    """Elimina guiones, espacios y convierte a mayusculas."""
    return re.sub(r'[\s\-]', '', plate).upper()


def similarity(a: str, b: str) -> float:
    """Porcentaje de caracteres correctos (longitud del menor / mayor)."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    # Longitud comun
    common = sum(1 for i in range(min(len(na), len(nb))) if na[i] == nb[i])
    return common / max(len(na), len(nb))


def main():
    print("=" * 100)
    print("COMPARACION: OCR sin sharpen vs con sharpen vs ground truth")
    print("=" * 100)
    print()

    total_correct_ns = 0
    total_correct_sh = 0
    total_plates = 0

    for video in ["VID2EDIT", "VID4EDIT", "VID7EDIT"]:
        gt_plates = FILE_TO_GT_PLATE[video]
        ns_results = OCR_NO_SHARPEN[video]
        sh_results = OCR_WITH_SHARPEN[video]

        print(f"\n### {video}")
        print(f"{'Archivo':<25s} {'GT':<12s} {'Sin sharpen':<15s} {'Conf':<6s} {'Match':<6s} {'Con sharpen':<15s} {'Conf':<6s} {'Match':<6s} {'Ganador':<10s}")
        print("-" * 100)

        for fname in sorted(gt_plates.keys()):
            gt = gt_plates[fname]
            ns = ns_results.get(fname, "?")
            sh = sh_results.get(fname, "?")

            ns_norm = normalize(ns)
            sh_norm = normalize(sh)
            gt_norm = normalize(gt)

            ns_match = "OK" if ns_norm == gt_norm else ""
            sh_match = "OK" if sh_norm == gt_norm else ""

            ns_sim = similarity(ns, gt)
            sh_sim = similarity(sh, gt)

            if ns_match == "OK":
                total_correct_ns += 1
            if sh_match == "OK":
                total_correct_sh += 1
            total_plates += 1

            # Determine winner
            if ns_match == "OK" and sh_match != "OK":
                winner = "Sin sharpen"
            elif sh_match == "OK" and ns_match != "OK":
                winner = "Con sharpen"
            elif ns_sim > sh_sim:
                winner = "Sin sharpen"
            elif sh_sim > ns_sim:
                winner = "Con sharpen"
            else:
                winner = "Empate"

            print(f"{fname:<25s} {gt:<12s} {ns:<15s} {'':6s} {ns_match:<6s} {sh:<15s} {'':6s} {sh_match:<6s} {winner}")

        print()

    print("=" * 100)
    print("RESUMEN")
    print("=" * 100)
    print(f"Sin sharpen: {total_correct_ns}/{total_plates} exact match ({total_correct_ns/total_plates*100:.1f}%)")
    print(f"Con sharpen: {total_correct_sh}/{total_plates} exact match ({total_correct_sh/total_plates*100:.1f}%)")


if __name__ == "__main__":
    main()
