#!/usr/bin/env bash
# InfractiVision ONLINE installer - macOS (per-user /Applications fallback)
set -euo pipefail
REPO="AbelMoyaICSI/InfractiVision"
BASE_URL="https://github.com/${REPO}/releases/latest/download"
PREFIX="/Applications/InfractiVision.app"
VARIANT="cpu"  # macOS siempre CPU (MPS futuro); CUDA no existe en Mac
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then TAG="Mac-arm64"; else TAG="Mac-x64"; fi

# Videos demo (deben coincidir con config/demo_videos.json)
DEMO_FILES=(
  "Av-Condorcanqui.mp4"
  "VID1EDIT ‐ Hecho con Clipchamp.mp4"
  "VID2COLISEO.MOV"
  "VID2EDIT ‐ Hecho con Clipchamp.mp4"
  "VID4EDIT ‐ Hecho con Clipchamp.mp4"
)
DEMO_URLS=(
  "https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/Av-Condorcanqui.mp4?alt=media&token=9ee9bd87-2a0f-4bf2-8acb-445f0bbb48e4"
  "https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/VID1EDIT%20%E2%80%90%20Hecho%20con%20Clipchamp.mp4?alt=media&token=b99a2f2d-a765-44bb-a4b4-63c2e8a1357a"
  "https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/VID2COLISEO.MOV?alt=media&token=10317415-ed30-4ae1-869f-3c47c31fdaa6"
  "https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/VID2EDIT%20%E2%80%90%20Hecho%20con%20Clipchamp.mp4?alt=media&token=9bcae3a5-b76a-4b70-ad5a-ea153cdaec18"
  "https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/VID4EDIT%20%E2%80%90%20Hecho%20con%20Clipchamp.mp4?alt=media&token=520a3110-d499-4a9e-b43d-cb054ca48e0a"
)

echo "== InfractiVision ONLINE installer (macOS) =="
echo "[*] Variante: $VARIANT ($TAG) -> $PREFIX"
# En macOS sin permisos usamos ~/Applications
if [[ ! -w "/Applications" ]]; then PREFIX="$HOME/Applications/InfractiVision.app"; fi
artifact="InfractiVision-${VARIANT}-${TAG}.zip"
url="${BASE_URL}/${artifact}"
echo "[*] Descargando $url"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
zip="$tmp/iv.zip"
if command -v curl >/dev/null 2>&1; then curl -fL --progress-bar "$url" -o "$zip"
else wget -O "$zip" "$url"; fi
echo "[*] Extrayendo a $PREFIX"
mkdir -p "$(dirname "$PREFIX")"
if command -v unzip >/dev/null 2>&1; then unzip -oq "$zip" -d "$(dirname "$PREFIX")"
else python3 -m zipfile -e "$zip" "$(dirname "$PREFIX")"; fi
# Quarantine remove si Gatekeeper bloquea sin firma
xattr -dr com.apple.quarantine "$PREFIX" 2>/dev/null || true

# Videos demo (descarga al directorio de datos del usuario, que es donde
# el exe empaquetado los busca; el runtime también reintenta)
VIDEO_BASE="$HOME/Library/Application Support/InfractiVision"
mkdir -p "$VIDEO_BASE/videos"
for i in "${!DEMO_FILES[@]}"; do
  f="${DEMO_FILES[$i]}"; u="${DEMO_URLS[$i]}"
  if [[ -f "$VIDEO_BASE/videos/$f" ]]; then continue; fi
  echo "[*] Descargando video demo: $f"
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 --progress-bar "$u" -o "$VIDEO_BASE/videos/$f"
  elif command -v wget >/dev/null 2>&1; then wget -O "$VIDEO_BASE/videos/$f" "$u"
  else python3 - "$u" "$VIDEO_BASE/videos/$f" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
  fi
done

echo "✓ Instalado en $PREFIX"
echo "  Ejecuta desde Launchpad o: open \"$PREFIX\""
echo "  Desinstalar: rm -rf \"$PREFIX\""
# Nota firma
cat <<NOTE

Nota firma: esta build no esta notarizada (sin Apple Developer cert).
Si macOS dice "no se puede verificar", ejecuta:
  xattr -dr com.apple.quarantine "$PREFIX"
o click derecho > Abrir.

NOTE
