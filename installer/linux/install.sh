#!/usr/bin/env bash
# InfractiVision ONLINE installer - Linux (per-user, XDG)
# Uso: curl -fsSL https://github.com/AbelMoyaICSI/InfractiVision/releases/latest/download/install.sh | bash
#   o: bash installer/linux/install.sh [--cpu|--cuda|--auto] [--prefix ~/.local/share/InfractiVision]
set -euo pipefail

REPO="AbelMoyaICSI/InfractiVision"
BASE_URL="https://github.com/${REPO}/releases/latest/download"
PREFIX="${HOME}/.local/share/InfractiVision"
VARIANT="auto"
DEMO="yes"
ARCH="$(uname -m)"
OS_TAG="Linux"
if [[ "$ARCH" == "x86_64" || "$ARCH" == "amd64" ]]; then ARCH="x64"
elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then ARCH="arm64"
fi

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

usage(){ echo "Usage: $0 [--auto|--cpu|--cuda] [--with-demo|--no-demo] [--prefix DIR]"; exit 0; }
while [[ $# -gt 0 ]]; do case "$1" in
  --cpu) VARIANT="cpu"; shift;;
  --cuda) VARIANT="cuda"; shift;;
  --auto) VARIANT="auto"; shift;;
  --with-demo) DEMO="yes"; shift;;
  --no-demo) DEMO="no"; shift;;
  --prefix) PREFIX="$2"; shift 2;;
  -h|--help) usage;;
  *) echo "Unknown arg $1"; usage;;
esac; done

has_nvidia_gpu(){
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then return 0; fi
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qi nvidia; then return 0; fi
  return 1
}

check_deps(){
  local missing=()
  if ! command -v python3 >/dev/null 2>&1; then missing+=("python3"); fi
  if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "[!] tkinter no disponible. Instala: sudo apt install python3-tk  (Debian/Ubuntu) | sudo dnf install python3-tkinter (Fedora) | sudo pacman -S tk (Arch)"
    missing+=("python3-tk")
  fi
  if ! ldconfig -p 2>/dev/null | grep -q libGL; then
    echo "[!] libGL no encontrada (necesaria para opencv). Instala: sudo apt install libgl1"
  fi
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[!] Dependencias faltantes: ${missing[*]}. El instalador continuara, pero la app puede fallar."
  fi
}

resolve_variant(){
  if [[ "$VARIANT" == "auto" ]]; then
    if has_nvidia_gpu; then echo "cuda"; else echo "cpu"; fi
  else echo "$VARIANT"; fi
}

download_demo_videos(){
  local dir="$1/videos"
  mkdir -p "$dir"
  local ok=0 fail=0
  for i in "${!DEMO_FILES[@]}"; do
    local f="${DEMO_FILES[$i]}" u="${DEMO_URLS[$i]}"
    if [[ -f "$dir/$f" ]]; then ok=$((ok+1)); continue; fi
    echo "[*] Descargando video demo: $f"
    if command -v curl >/dev/null 2>&1; then
      curl -fL --retry 3 --progress-bar "$u" -o "$dir/$f" && ok=$((ok+1)) || fail=$((fail+1))
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$dir/$f" "$u" && ok=$((ok+1)) || fail=$((fail+1))
    elif command -v python3 >/dev/null 2>&1; then
      python3 - "$u" "$dir/$f" <<'PY' && ok=$((ok+1)) || fail=$((fail+1))
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
    else
      echo "[!] Sin curl/wget/python3 para descargar $f"; fail=$((fail+1))
    fi
  done
  echo "✓ Videos demo: $ok ok, $fail fallidos (la app reintenta al primer inicio si faltan)"
}

main(){
  echo "== InfractiVision ONLINE installer (Linux) =="
  check_deps
  local var; var="$(resolve_variant)"
  echo "[*] Variante detectada: $var (arch=$ARCH, prefix=$PREFIX)"
  local artifact="InfractiVision-${var}-${OS_TAG}-${ARCH}.zip"
  local url="${BASE_URL}/${artifact}"
  echo "[*] Descargando $url"
  local tmp; tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  local zip="$tmp/iv.zip"
  if command -v curl >/dev/null 2>&1; then curl -fL --progress-bar "$url" -o "$zip"
  elif command -v wget >/dev/null 2>&1; then wget -O "$zip" "$url"
  else echo "Necesitas curl o wget"; exit 1; fi
  echo "[*] Extrayendo a $PREFIX"
  mkdir -p "$PREFIX"
  if command -v unzip >/dev/null 2>&1; then unzip -oq "$zip" -d "$PREFIX"
  else python3 -m zipfile -e "$zip" "$PREFIX"; fi
  chmod +x "$PREFIX/InfractiVision" 2>/dev/null || true
  # Videos demo (descarga al directorio de datos del usuario, que es donde
  # el exe empaquetado los busca; el runtime también reintenta)
  if [[ "$DEMO" == "yes" ]]; then
    local cfg_base="${XDG_CONFIG_HOME:-$HOME/.config}/InfractiVision"
    download_demo_videos "$cfg_base"
  else
    echo "[*] Videos demo omitidos (--no-demo)"
  fi
  # .desktop
  local app_dir="${HOME}/.local/share/applications"
  mkdir -p "$app_dir"
  cat > "$app_dir/infractivision.desktop" <<DESKTOP
[Desktop Entry]
Name=InfractiVision
Comment=Sistema de deteccion de infracciones de trafico
Exec=${PREFIX}/InfractiVision
Icon=${PREFIX}/img/icon.ico
Terminal=false
Type=Application
Categories=Science;
DESKTOP
  update-desktop-database "$app_dir" 2>/dev/null || true
  echo ""
  echo "✓ Instalado en $PREFIX"
  echo "  Ejecuta: $PREFIX/InfractiVision"
  echo "  O busca 'InfractiVision' en tu menu"
  echo "  Desinstalar: rm -rf $PREFIX $app_dir/infractivision.desktop"
  # Probe GPU runtime info
  if has_nvidia_gpu; then echo "  GPU: NVIDIA detectada -> variante cuda (si falla driver, app hace fallback a CPU)"
  else echo "  GPU: No detectada -> variante cpu"; fi
}

main "$@"
