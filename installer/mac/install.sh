#!/usr/bin/env bash
# InfractiVision ONLINE installer - macOS (per-user /Applications fallback)
set -euo pipefail
REPO="AbelMoyaICSI/InfractiVision"
BASE_URL="https://github.com/${REPO}/releases/latest/download"
PREFIX="/Applications/InfractiVision.app"
VARIANT="cpu"  # macOS siempre CPU (MPS futuro); CUDA no existe en Mac
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then TAG="Mac-arm64"; else TAG="Mac-x64"; fi

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
