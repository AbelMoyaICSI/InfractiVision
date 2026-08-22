#!/usr/bin/env bash
# build-pkg.sh — Genera InfractiVision-Setup-Online.pkg y .dmg (requiere Xcode)
# Uso: bash installer/mac/build-pkg.sh --version 2.1.0
set -euo pipefail
VERSION="${1:-2.1.0}"; [[ "$1" == "--version" ]] && VERSION="$2"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$ROOT/dist/mac-pkg"
APP_NAME="InfractiVision"
IDENTIFIER="com.infractivision.app"

echo "[mac-pkg] version=$VERSION"
rm -rf "$STAGE"; mkdir -p "$STAGE/scripts" "$STAGE/root/Applications"

# preinstall/postinstall scripts que descargan artefacto online
cat > "$STAGE/scripts/postinstall" <<'POST'
#!/bin/bash
set -e
REPO="AbelMoyaICSI/InfractiVision"
BASE_URL="https://github.com/${REPO}/releases/latest/download"
ARCH="$(uname -m)"; [[ "$ARCH" == "arm64" ]] && TAG="Mac-arm64" || TAG="Mac-x64"
URL="${BASE_URL}/InfractiVision-cpu-${TAG}.zip"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "[postinstall] Descargando $URL"
curl -fL --progress-bar "$URL" -o "$TMP/iv.zip"
echo "[postinstall] Extrayendo a /Applications/InfractiVision.app"
mkdir -p /Applications
unzip -oq "$TMP/iv.zip" -d /Applications
xattr -dr com.apple.quarantine /Applications/InfractiVision.app 2>/dev/null || true
exit 0
POST
chmod +x "$STAGE/scripts/postinstall"
touch "$STAGE/scripts/preinstall" && chmod +x "$STAGE/scripts/preinstall"

# pkgbuild (sin payload, solo scripts)
pkgbuild --identifier "$IDENTIFIER" --version "$VERSION" --scripts "$STAGE/scripts" --root "$STAGE/root" "$ROOT/dist/InfractiVision-Setup-Online-${VERSION}.pkg"
echo "[mac-pkg] pkg: $ROOT/dist/InfractiVision-Setup-Online-${VERSION}.pkg"

# dmg opcional (requiere create-dmg)
if command -v create-dmg >/dev/null 2>&1; then
  create-dmg --volname "InfractiVision $VERSION" --volicon "$ROOT/img/icon.ico" --window-pos 200 120 --window-size 600 400 --icon-size 100 --icon "InfractiVision-Setup-Online-${VERSION}.pkg" 150 150 --hide-extension "InfractiVision-Setup-Online-${VERSION}.pkg" --app-drop-link 450 150 "$ROOT/dist/InfractiVision-Setup-Online-${VERSION}.dmg" "$STAGE" || true
  echo "[mac-pkg] dmg: $ROOT/dist/InfractiVision-Setup-Online-${VERSION}.dmg"
else
  echo "[mac-pkg] create-dmg no instalado, pkg listo (instala dmg con: brew install create-dmg)"
fi

# Codesign si hay cert
if security find-identity -v -p codesigning 2>/dev/null | grep -q "Developer ID"; then
  echo "[mac-pkg] Firmando pkg..."
  productsign --sign "Developer ID Installer" "$ROOT/dist/InfractiVision-Setup-Online-${VERSION}.pkg" "$ROOT/dist/InfractiVision-Setup-Online-${VERSION}-signed.pkg" || true
fi
