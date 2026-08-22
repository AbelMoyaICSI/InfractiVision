# InfractiVision - Instalador ONLINE

## Resumen
Instalador 100% online por SO con deteccion automatica de GPU.

| SO | Instalador | Tamaño stub | Que descarga | GPU |
|---|---|---|---|---|
| Windows 10+ | `InfractiVision-Setup-Online.exe` (Inno Setup 6) | ~8 MB | `InfractiVision-cpu-Win-x64.zip` (900MB) o `cuda-Win-x64.zip` (1.4GB) | Detecta `nvidia-smi` / `wmic` → CUDA else CPU |
| Linux | `installer/linux/install.sh` | 4 KB | `InfractiVision-cpu/cuda-Linux-x64.zip` | `lspci` + `nvidia-smi -L` |
| macOS | `installer/mac/install.sh` / `.pkg` | 3 KB | `InfractiVision-cpu-Mac-x64/arm64.zip` | Siempre CPU (CUDA no existe) |

Runtime siempre hace fallback: `src/core/ocr/lprnet_engine.py:87` `torch.cuda.is_available()` → si stub eligio mal, la app corre en CPU.

## Uso usuario final (sin compilar)

### Windows
1. Descarga `InfractiVision-Setup-Online.exe` desde Releases
2. Doble click → elige carpeta (default `%APPDATA%\InfractiVision`) → descarga automatica (requiere internet)
3. Si falta VC++ Redist, el instalador avisa con link a `https://aka.ms/vs/17/release/vc_redist.x64.exe`

### Linux
```bash
curl -fsSL https://github.com/AbelMoyaICSI/InfractiVision/releases/latest/download/install.sh | bash
# o local:
bash installer/linux/install.sh --auto
bash installer/linux/install.sh --cpu --prefix ~/.local/share/InfractiVision
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
- `release.yml`: on tag `v*` construye 5 artefactos (Win cpu/cuda, Linux cpu/cuda, Mac cpu) + `Setup-Online.exe` + publica Release.
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
