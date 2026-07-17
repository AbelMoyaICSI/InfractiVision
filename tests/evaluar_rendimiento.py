#!/usr/bin/env python
"""
evaluar_rendimiento.py — evaluación de rendimiento de InfractiVision
contra ground truth de tests/verdad.test.json.

Por cada video del dataset de verdad:
  1. Crea una config CLIInfractionPipeline con el polígono y semáforo de verdad
  2. Ejecuta pipeline.process(video_path) → captura NID, NIE, placas, tiempos
  3. Compara infracciones detectadas, placas reconocidas y tiempos vs verdad terreno
  4. Genera tabla comparativa, JSON y TXT de resumen

Uso:
    python tests/evaluar_rendimiento.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapter.process_video import CLIInfractionPipeline


VERDAD_PATH = PROJECT_ROOT / "tests" / "verdad.test.json"
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluacion"
SEPARATOR_WIDTH = 140


def normalizar_placa(placa: str) -> str:
    return placa.replace("-", "").replace(" ", "").upper()


def distancia_levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _match_placas(
    placas_detectadas: list[str],
    placas_esperadas: list[str],
) -> dict:
    normalizadas_det = [normalizar_placa(p) for p in placas_detectadas]
    normalizadas_esp = [normalizar_placa(p) for p in placas_esperadas]

    matched_exact: list[str] = []
    matched_fuzzy: list[tuple[str, str, int]] = []
    no_match: list[str] = []
    usadas = set()

    for det_raw, det_norm in zip(placas_detectadas, normalizadas_det):
        best_esp = None
        best_dist = 999
        best_idx = -1

        for i, (esp_raw, esp_norm) in enumerate(zip(placas_esperadas, normalizadas_esp)):
            if i in usadas:
                continue
            if det_norm == esp_norm:
                best_esp = esp_raw
                best_idx = i
                best_dist = 0
                break
            dist = distancia_levenshtein(det_norm, esp_norm)
            if dist <= 2 and dist < best_dist:
                best_esp = esp_raw
                best_dist = dist
                best_idx = i

        if best_dist == 0 and best_esp is not None:
            matched_exact.append(best_esp)
            usadas.add(best_idx)
        elif best_dist <= 2 and best_esp is not None:
            matched_fuzzy.append((det_raw, best_esp, best_dist))
            usadas.add(best_idx)
        else:
            no_match.append(det_raw)

    no_detectadas = [
        esp_raw
        for i, (esp_raw, _) in enumerate(zip(placas_esperadas, normalizadas_esp))
        if i not in usadas
    ]

    return {
        "matched_exact": matched_exact,
        "matched_fuzzy": matched_fuzzy,
        "no_match": no_match,
        "no_detectadas": no_detectadas,
        "total_esperado": len(placas_esperadas),
        "total_detectado": len(placas_detectadas),
        "total_matched": len(matched_exact) + len(matched_fuzzy),
    }


def _build_config(entry: dict) -> dict:
    return {
        "polygon": entry["polygon"],
        "semaphore": {
            "green": entry["green"],
            "yellow": entry["yellow"],
            "red": entry["red"],
            "start_offset_seconds": 0.0,
        },
        "conf_vehicle": 0.50,
        "conf_plate": 0.40,
        "batch_size": 4,
        "rectification": True,
        "avenue": "Evaluación",
        "time_slot": "Evaluación",
    }


def _pct(valor: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (valor / base) * 100.0


def _pct_str(valor: float, base: float) -> str:
    return f"{_pct(valor, base):.1f}%"


def _truncar(nombre: str, largo: int = 36) -> str:
    if len(nombre) <= largo:
        return nombre.ljust(largo)
    return nombre[: largo - 3] + "..."


def main() -> int:
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("  EVALUACIÓN DE RENDIMIENTO — InfractiVision vs Ground Truth")
    print("=" * SEPARATOR_WIDTH)

    if not VERDAD_PATH.exists():
        print(f"❌ No existe {VERDAD_PATH}")
        return 1

    with open(str(VERDAD_PATH), "r", encoding="utf-8") as f:
        verdad = json.load(f)

    entries = verdad.get("videos_verdad", [])
    if not entries:
        print("❌ No hay entradas en verdad.test.json")
        return 1

    print(f"\n📂 Videos en dataset: {len(entries)}")
    print(f"📂 Videos en disco: {len(os.listdir(str(VIDEOS_DIR)))}")

    resultados: list[dict] = []
    acumulado = {
        "infr_esp": 0,
        "infr_det": 0,
        "infr_able_esp": 0,
        "nid": 0,
        "nie": 0,
        "placas_esp": 0,
        "placas_matched": 0,
        "tiempo_total": 0.0,
        "videos_procesados": 0,
    }

    for idx, entry in enumerate(entries, 1):
        path_name = entry["path_name"]
        video_path = VIDEOS_DIR / path_name

        print(f"\n{'─' * SEPARATOR_WIDTH}")
        print(f"  [{idx}/{len(entries)}] {path_name}")

        if not video_path.exists():
            print(f"  ⏭️  SKIP: no existe en videos/")
            continue

        config = _build_config(entry)
        infraccione_esp = entry.get("infraccione", 0)
        able_detec_esp = entry.get("ingfraccion_able_detec", 0)
        placas_esp = entry.get("cars", [])

        try:
            pipeline = CLIInfractionPipeline(config, use_new=False)
            result = pipeline.process(str(video_path))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        nid = result["nid_count"]
        nie = result["nie_count"]
        infr_det = nid + nie

        placas_det = [
            inf["plate"]
            for inf in result["infractions"]
            if inf["clasificacion"] == "NID" and inf["plate"]
            and not inf["plate"].startswith("NIE_")
        ]

        match_result = _match_placas(placas_det, placas_esp)

        pct_infr = _pct(infr_det, infraccione_esp)
        pct_able = _pct(infr_det, able_detec_esp)
        pct_placas = _pct(match_result["total_matched"], match_result["total_esperado"])

        tiempo = result.get("elapsed_seconds", 0.0)
        tiempo_video = entry.get("time", 0)

        pct_global = (
            (pct_infr + pct_able + pct_placas) / 3.0
            if infraccione_esp > 0
            else _pct(infr_det, able_detec_esp)
        )

        print(f"  📊 Infracciones: esperado={infraccione_esp} (detectables={able_detec_esp}) → "
              f"detectado={infr_det} (NID={nid}, NIE={nie})")
        print(f"  📊 Placas: esperado={match_result['total_esperado']} → "
              f"matched={match_result['total_matched']} "
              f"(exactos={len(match_result['matched_exact'])}, "
              f"fuzzy={len(match_result['matched_fuzzy'])})")
        print(f"  ⏱️  Tiempo pipeline: {tiempo:.1f}s (video: {tiempo_video}s)")
        print(f"  🎯 %Acierto infracciones: {pct_infr:.1f}% | "
              f"detectables: {pct_able:.1f}% | "
              f"placas: {pct_placas:.1f}%")

        resultado = {
            "video": path_name,
            "infraccione_esperado": infraccione_esp,
            "infraccione_detectable": able_detec_esp,
            "nid": nid,
            "nie": nie,
            "infr_detectado": infr_det,
            "pct_infracciones": round(pct_infr, 1),
            "pct_detectables": round(pct_able, 1),
            "placas_esperado": match_result["total_esperado"],
            "placas_matched": match_result["total_matched"],
            "placas_exactas": len(match_result["matched_exact"]),
            "placas_fuzzy": len(match_result["matched_fuzzy"]),
            "placas_no_match": match_result["no_match"],
            "placas_no_detectadas": match_result["no_detectadas"],
            "pct_placas": round(pct_placas, 1),
            "pct_global": round(pct_global, 1),
            "tiempo_pipeline_s": round(tiempo, 1),
            "tiempo_video_s": tiempo_video,
            "elapsed_seconds": tiempo,
        }
        resultados.append(resultado)

        acumulado["infr_esp"] += infraccione_esp
        acumulado["infr_det"] += infr_det
        acumulado["infr_able_esp"] += able_detec_esp
        acumulado["nid"] += nid
        acumulado["nie"] += nie
        acumulado["placas_esp"] += match_result["total_esperado"]
        acumulado["placas_matched"] += match_result["total_matched"]
        acumulado["tiempo_total"] += tiempo
        acumulado["videos_procesados"] += 1

    n_videos = acumulado["videos_procesados"]
    if n_videos == 0:
        print("\n❌ Ningún video procesado.")
        return 1

    pct_infr_tot = _pct(acumulado["infr_det"], acumulado["infr_esp"])
    pct_able_tot = _pct(acumulado["infr_det"], acumulado["infr_able_esp"])
    pct_placas_tot = _pct(acumulado["placas_matched"], acumulado["placas_esp"])
    pct_global_tot = round(
        (sum(r["pct_global"] for r in resultados) / n_videos), 1
    ) if n_videos > 0 else 0.0

    tir = acumulado["nid"] + acumulado["nie"]
    ti_pct = (acumulado["nid"] / tir * 100) if tir > 0 else 0.0

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("  TABLA COMPARATIVA — RESULTADOS POR VIDEO")
    print("=" * SEPARATOR_WIDTH)

    header = (
        f"  {'Video':<38s} │ {'Infr. Det.':>22s} │ {'Placas':>25s} │ {'Tiempo (s)':>14s} │ {'% Acierto':>10s}"
    )
    subhdr = (
        f"  {'':38s} │ {'Esp  Det   Rec%':>22s} │ {'Esp  Mat   Rec%':>25s} │ {'Pipe   Video':>14s} │ {'Global':>10s}"
    )

    print(header)
    print(subhdr)
    print("─" * SEPARATOR_WIDTH)

    for r in resultados:
        infr_col = f"{r['infraccione_esperado']:>4} {r['infr_detectado']:>4} {r['pct_infracciones']:>6.1f}%"
        placas_col = f"{r['placas_esperado']:>4} {r['placas_matched']:>4} {r['pct_placas']:>6.1f}%"
        tiempo_col = f"{r['tiempo_pipeline_s']:>5.0f}  {r['tiempo_video_s']:>5.0f}"
        acierto_col = f"{r['pct_global']:>6.1f}%"
        print(
            f"  {_truncar(r['video'])} │ {infr_col} │ {placas_col} │ {tiempo_col} │ {acierto_col}"
        )

    print("─" * SEPARATOR_WIDTH)

    infr_tot_col = f"{acumulado['infr_esp']:>4} {acumulado['infr_det']:>4} {pct_infr_tot:>6.1f}%"
    placas_tot_col = f"{acumulado['placas_esp']:>4} {acumulado['placas_matched']:>4} {pct_placas_tot:>6.1f}%"
    tiempo_tot_col = f"{acumulado['tiempo_total']:>5.0f}  {'—':>5s}"
    acierto_tot_col = f"{pct_global_tot:>6.1f}%"
    print(
        f"  {'TOTALES':<38s} │ {infr_tot_col} │ {placas_tot_col} │ {tiempo_tot_col} │ {acierto_tot_col}"
    )

    print("─" * SEPARATOR_WIDTH)
    print(f"\n  INDICADORES GLOBALES")
    print(f"  NID: {acumulado['nid']} detectados")
    print(f"  NIE: {acumulado['nie']} detectados")
    print(f"  TIR: {tir} (NID + NIE)")
    print(f"  TI:  {ti_pct:.1f}% (NID / TIR × 100)")
    print(f"  TR:  {acumulado['tiempo_total']:.1f}s total ({acumulado['tiempo_total'] / 60:.1f} min)")
    print(f"\n  Referencia verdad.test.json (valores globales):")
    print(f"  NID: {verdad.get('NID', '?')}  |  NIE: {verdad.get('NIE', '?')}")
    print(f"  TI:  {verdad.get('TI', '?')}%  |  TR:  {verdad.get('TR', '?')}s")
    print("=" * SEPARATOR_WIDTH)

    reporte = {
        "fecha_evaluacion": time.strftime("%d/%m/%Y %H:%M:%S"),
        "dataset": str(VERDAD_PATH),
        "videos_procesados": n_videos,
        "totales": {
            "infracciones_esperado": acumulado["infr_esp"],
            "infracciones_detectado": acumulado["infr_det"],
            "pct_infracciones": round(pct_infr_tot, 1),
            "infracciones_detectables_esperado": acumulado["infr_able_esp"],
            "pct_detectables": round(pct_able_tot, 1),
            "placas_esperado": acumulado["placas_esp"],
            "placas_matched": acumulado["placas_matched"],
            "pct_placas": round(pct_placas_tot, 1),
            "nid": acumulado["nid"],
            "nie": acumulado["nie"],
            "tir": tir,
            "ti_pct": round(ti_pct, 1),
            "tiempo_total_s": round(acumulado["tiempo_total"], 1),
            "pct_global_promedio": pct_global_tot,
        },
        "referencia_verdad": {
            "NID": verdad.get("NID"),
            "NIE": verdad.get("NIE"),
            "TI": verdad.get("TI"),
            "TR": verdad.get("TR"),
        },
        "resultados_por_video": resultados,
    }

    json_path = OUTPUT_DIR / "evaluacion.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Reporte JSON: {json_path}")

    txt_path = OUTPUT_DIR / "resumen.txt"
    with open(str(txt_path), "w", encoding="utf-8") as f:
        f.write("EVALUACIÓN DE RENDIMIENTO — InfractiVision vs Ground Truth\n")
        f.write("=" * SEPARATOR_WIDTH + "\n\n")
        f.write(f"Fecha: {reporte['fecha_evaluacion']}\n")
        f.write(f"Dataset: {reporte['dataset']}\n")
        f.write(f"Videos procesados: {n_videos}\n\n")

        f.write("TABLA COMPARATIVA POR VIDEO\n")
        f.write("-" * SEPARATOR_WIDTH + "\n")
        f.write(f"{'Video':<38s} | {'Infr. Det.':>22s} | {'Placas':>25s} | {'Tiempo':>14s} | %Acierto\n")
        f.write(f"{'':38s} | {'Esp  Det   Rec%':>22s} | {'Esp  Mat   Rec%':>25s} | {'Pipe   Video':>14s} | Global\n")
        f.write("-" * SEPARATOR_WIDTH + "\n")

        for r in resultados:
            infr_col = f"{r['infraccione_esperado']:>4} {r['infr_detectado']:>4} {r['pct_infracciones']:>6.1f}%"
            placas_col = f"{r['placas_esperado']:>4} {r['placas_matched']:>4} {r['pct_placas']:>6.1f}%"
            tiempo_col = f"{r['tiempo_pipeline_s']:>5.0f}  {r['tiempo_video_s']:>5.0f}"
            acierto_col = f"{r['pct_global']:>6.1f}%"
            f.write(f"{_truncar(r['video'])} | {infr_col} | {placas_col} | {tiempo_col} | {acierto_col}\n")

        f.write("-" * SEPARATOR_WIDTH + "\n")
        f.write(f"{'TOTALES':<38s} | {infr_tot_col} | {placas_tot_col} | {tiempo_tot_col} | {acierto_tot_col}\n\n")

        f.write("DETALLE DE PLACAS POR VIDEO\n")
        f.write("-" * SEPARATOR_WIDTH + "\n")
        for r in resultados:
            f.write(f"\n{r['video']}\n")
            f.write(f"  Esperadas ({r['placas_esperado']}):\n")
            f.write(f"  Matched exactas ({r['placas_exactas']}):\n")
            f.write(f"  Matched fuzzy ({r['placas_fuzzy']}):\n")
            if r["placas_no_match"]:
                f.write(f"  No match (detectadas extra): {r['placas_no_match']}\n")
            if r["placas_no_detectadas"]:
                f.write(f"  No detectadas: {r['placas_no_detectadas']}\n")

        f.write("\nINDICADORES GLOBALES\n")
        f.write(f"  NID: {acumulado['nid']}\n")
        f.write(f"  NIE: {acumulado['nie']}\n")
        f.write(f"  TIR: {tir}\n")
        f.write(f"  TI:  {ti_pct:.1f}%\n")
        f.write(f"  TR:  {acumulado['tiempo_total']:.1f}s\n")

    print(f"📄 Reporte TXT:  {txt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
