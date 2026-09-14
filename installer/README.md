# InfractiVision - Instalador ONLINE single-file (build unico CUDA con fallback CPU)

## Resumen
Build unico compilado con CUDA (`requirements.txt` torch 2.8.0+cu128). Si hay NVIDIA usa GPU, si no hay corre en CPU (`torch.cuda.is_available()`). Sin versiones CPU/CUDA separadas, sin pip on-demand, sin checkbox.

| SO | Instalador | Que descarga | GPU |
|---|---|---|---|
| Windows 10+ | `InfractiVision-Setup-Online.exe` (Inno Setup 6) | ONEDIR CUDA embebido + 5 videos demo | **Ventana informativa**: `nvidia-smi` → `Get-CimInstance` → `wmic` → `✅ usará GPU` o `❌ correrá en CPU` (mismo binario) |
| Linux | `installer/linux/install.sh` | `InfractiVision-cuda-Linux-x64.zip` + 5 videos demo | **Informativo**: `nvidia-smi`/`lspci` → mensaje `usará GPU` o `correrá en CPU` |
| macOS | `installer/mac/install.sh` / `.pkg` | `InfractiVision-cpu-Mac-x64/arm64.zip` + 5 videos demo | Siempre CPU (macOS no tiene CUDA, build legacy CPU) |

Los 5 videos demo se descargan al **directorio de datos del usuario**:
- Win: `%APPDATA%\InfractiVision\videos`
- Linux: `$XDG_CONFIG_HOME/InfractiVision/videos` (default `~/.config/InfractiVision/videos`)
- macOS: `~/Library/Application Support/InfractiVision/videos`

Esa es la carpeta `videos/` que el exe busca (persistente; `_MEIPASS` es temporal y no sirve). Si falla la red al instalar, la app **reintenta la descarga al primer inicio** (`src/infrastructure/storage/demo_video_downloader.py`, botón "⬇️ Descargar Demo" en el selector de videos). Videos y presets vienen del manifest `config/demo_videos.json` (hashes sha256 verificados).

Runtime siempre hace fallback: `src/core/detection/vehicle_detector.py:26`, `plate_detector.py:64` `torch.cuda.is_available()` → el mismo binario corre en CPU si no hay GPU.

## Secretos incluidos en el artefacto
El `.exe` empaqueta (solo si existen al compilar):
- `infractivision-e8c03-firebase-adminsdk-fbsvc-957f584093.json` — Service Account para **migraciones Firestore desde el exe instalado** (`src/automations/firestore_migrator.py`).
- `.env` — token `PLATE_RECOGNIZER_API_TOKEN` para **validación cloud de placas** desde el exe (`src/infrastructure/ocr/cloud_plate_readers.py`).

> ⚠️ **Advertencia de seguridad**: un onefile de PyInstaller es un zip; cualquiera con el `.exe` puede extraer esos secretos. Rotar la Service Account / el token de Plate Recognizer exige re-build + re-release. No subas el `Setup-Online.exe` ni los zips con estos secretos a un canal público salvo que asumas ese riesgo.

Como fallback, la app también lee el token desde `APPDATA_DIR/plate_recognizer.json` (formato `{"token": "..."}`) y la Service Account desde `APPDATA_DIR/`.

## Uso usuario final (sin compilar)

### Windows
1. Descarga `InfractiVision-Setup-Online.exe` desde Releases
2. Doble click → **ventana "Detección de hardware"** muestra `✅ GPU NVIDIA ... — usará GPU` o `❌ No detectada → correrá en CPU` (informativo, mismo binario) → Next → elige carpeta (default `%APPDATA%\InfractiVision`)
3. Si falta VC++ Redist, el instalador avisa con link a `https://aka.ms/vs/17/release/vc_redist.x64.exe`

### Linux
```bash
curl -fsSL https://github.com/AbelMoyaICSI/InfractiVision/releases/latest/download/install.sh | bash
# o local:
bash installer/linux/install.sh --prefix ~/.local/share/InfractiVision
bash installer/linux/install.sh --no-demo   # sin descargar videos demo
# Flags legacy --cpu/--cuda/--with-cuda-pip/--no-cuda-pip se aceptan como no-op
```

### macOS
```bash
bash installer/mac/install.sh
# si Gatekeeper bloquea (sin firma):
xattr -dr com.apple.quarantine /Applications/InfractiVision.app
```

## Uso desarrollador (generar artefactos)

```bash
# 1. Build local canonico (requiere requirements.txt con CUDA)
pip install -r requirements.txt
python scripts/build_online.py --variant cuda
python scripts/build_online.py --variant cuda --zip  # + zip para Releases (Linux: InfractiVision-cuda-Linux-x64.zip)
# legacy solo macOS:
pip install -r requirements-cpu.txt
python scripts/build_online.py --variant cpu --zip

# 2. Verificacion offline (sin red)
python scripts/verify_installer.py

# 3. Windows Setup (requiere Inno Setup 6: choco install innosetup)
iscc installer/win/online.iss
# genera dist/InfractiVision-Setup-Online.exe

# 4. macOS pkg/dmg (requiere Xcode)
bash installer/mac/build-pkg.sh --version 2.1.0
```

## CI/CD
- `release.yml`: on tag `v*` instala `requirements.txt` → construye ONEDIR CUDA (`InfractiVision-ONEDIR-CUDA.spec`) → `iscc online.iss` embebe con lzma2 → publica solo `InfractiVision-Setup-Online.exe`. Sin pip en el instalador.
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
  win/online.iss       # Inno Setup single-file CUDA unico (pagina GPU informativa)
  linux/install.sh     # bash per-user XDG (artefacto cuda unico)
  mac/install.sh       # macOS curl+unzip (legacy CPU)
  mac/build-pkg.sh     # pkgbuild + dmg
InfractiVision.spec          # spec ONLINE (sin videos/secrets/data)
InfractiVision-ONEDIR-CUDA.spec  # spec canonico (requirements.txt)
InfractiVision-ONEDIR-CPU.spec   # spec legacy macOS (requirements-cpu.txt)
scripts/build_online.py
scripts/verify_installer.py
```
