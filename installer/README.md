# InfractiVision - Instalador ONLINE (opcion 3 sin GCS)

## Resumen
Instalador 100% online por SO. Opcion 3: sin GCS, solo CPU (0 costo almacenamiento).

| SO | Instalador | Tamaño stub | Que descarga | GPU |
|---|---|---|---|---|
| Windows 10+ | `InfractiVision-Setup-Online.exe` (Inno Setup 6) | ~8 MB | `InfractiVision-cpu-Win-x64.zip` (900MB) + 5 videos demo | **Ventana GPU**: detecta `nvidia-smi` → `Get-CimInstance` → `wmic` → muestra `✅ NVIDIA ... (CPU forzado)` o `❌ → CPU`; siempre descarga CPU |
| Linux | `installer/linux/install.sh` | 4 KB | `InfractiVision-cpu-Linux-x64.zip` + 5 videos demo | Detecta pero fuerza CPU (cuda deshabilitado sin GCS) |
| macOS | `installer/mac/install.sh` / `.pkg` | 3 KB | `InfractiVision-cpu-Mac-x64/arm64.zip` + 5 videos demo | Siempre CPU |

Los 5 videos demo se descargan al **directorio de datos del usuario**:
- Win: `%APPDATA%\InfractiVision\videos`
- Linux: `$XDG_CONFIG_HOME/InfractiVision/videos` (default `~/.config/InfractiVision/videos`)
- macOS: `~/Library/Application Support/InfractiVision/videos`

Esa es la carpeta `videos/` que el exe busca (persistente; `_MEIPASS` es temporal y no sirve). Si falla la red al instalar, la app **reintenta la descarga al primer inicio** (`src/infrastructure/storage/demo_video_downloader.py`, botón "⬇️ Descargar Demo" en el selector de videos). Videos y presets vienen del manifest `config/demo_videos.json` (hashes sha256 verificados).

Runtime siempre hace fallback: `src/core/ocr/lprnet_engine.py:87` `torch.cuda.is_available()` → si stub eligio mal, la app corre en CPU.

## Secretos incluidos en el artefacto
El `.exe` empaqueta (solo si existen al compilar):
- `infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json` — Service Account para **migraciones Firestore desde el exe instalado** (`src/automations/firestore_migrator.py`).
- `.env` — token `PLATE_RECOGNIZER_API_TOKEN` para **validación cloud de placas** desde el exe (`src/infrastructure/ocr/cloud_plate_readers.py`).

> ⚠️ **Advertencia de seguridad**: un onefile de PyInstaller es un zip; cualquiera con el `.exe` puede extraer esos secretos. Rotar la Service Account / el token de Plate Recognizer exige re-build + re-release. No subas el `Setup-Online.exe` ni los zips con estos secretos a un canal público salvo que asumas ese riesgo.

Como fallback, la app también lee el token desde `APPDATA_DIR/plate_recognizer.json` (formato `{"token": "..."}`) y la Service Account desde `APPDATA_DIR/`.

## Uso usuario final (sin compilar)

### Windows
1. Descarga `InfractiVision-Setup-Online.exe` desde Releases
2. Doble click → **ventana "Detección de hardware"** muestra `🔍 Detectando...` → `✅ NVIDIA GeForce RTX ... detectada (CPU forzado)` o `❌ No detectada → CPU` y continúa solo → elige carpeta (default `%APPDATA%\InfractiVision`) → descarga automática CPU (900MB, sin GCS)
3. Si falta VC++ Redist, el instalador avisa con link a `https://aka.ms/vs/17/release/vc_redist.x64.exe`

### Linux
```bash
curl -fsSL https://github.com/AbelMoyaICSI/InfractiVision/releases/latest/download/install.sh | bash
# o local:
bash installer/linux/install.sh --auto
bash installer/linux/install.sh --cpu --prefix ~/.local/share/InfractiVision
bash installer/linux/install.sh --no-demo   # sin descargar videos demo
```

### macOS
```bash
bash installer/mac/install.sh
# si Gatekeeper bloquea (sin firma):
xattr -dr com.apple.quarantine /Applications/InfractiVision.app
```

## Uso desarrollador (generar artefactos)

```bash
# 1. Build local (requiere requirements instalados)
python scripts/build_online.py --variant cpu
python scripts/build_online.py --variant cuda
python scripts/build_online.py --variant all --zip  # genera zips en dist/

# 2. Verificacion offline (sin red)
python scripts/verify_installer.py

# 3. Windows Setup (requiere Inno Setup 6: choco install innosetup)
iscc installer/win/online.iss
# genera dist/InfractiVision-Setup-Online.exe

# 4. macOS pkg/dmg (requiere Xcode)
bash installer/mac/build-pkg.sh --version 2.1.0
```

## CI/CD
- `release.yml` (opcion 3 sin GCS): on tag `v*` construye solo `cpu` (`InfractiVision-ONEDIR-CPU.spec` + `requirements-cpu.txt`) → zip `InfractiVision-cpu-Win-x64.zip` + `Setup-Online.exe` con ventana GPU informativa + publica 2 artefactos (sin GCS, CUDA deshabilitada).
- `deps.yml`: verifica `requirements*.txt` con `scripts/ci_smoke_test.py`.

## Firma de codigo (diferida a v1.1)
- Windows: OV ~$300/año `signtool sign /tr http://timestamp.digicert.com ...`
- macOS: Apple Developer $99/año `codesign --hardened-runtime` + `notarytool`
- v1.0 sale sin firma con aviso SmartScreen/Gatekeeper "Mas informacion > Ejecutar de todas formas"

## Desinstalacion
- Win: Panel de control > InfractiVision > Desinstalar (borra `{app}`, conserva `%APPDATA%/InfractiVision/output` si quieres)
- Linux: `rm -rf ~/.local/share/InfractiVision ~/.local/share/applications/infractivision.desktop`
- Mac: `rm -rf /Applications/InfractiVision.app`

## Estructura
```
installer/
  win/online.iss       # Inno Setup online stub
  linux/install.sh     # bash per-user XDG
  mac/install.sh       # macOS curl+unzip
  mac/build-pkg.sh     # pkgbuild + dmg
InfractiVision.spec          # spec ONLINE (sin videos/secrets/data)
InfractiVision-CPU.spec
InfractiVision-CUDA.spec
scripts/build_online.py
scripts/verify_installer.py
```
