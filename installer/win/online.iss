; InfractiVision Setup Online - Windows single-file (opcion 3 sin GCS + CUDA via pip autoseleccionado)
; Modo: single-file 177M (lzma2) — embebe ONEDIR CPU completo, sin zip separado.
; - Detecta nvidia-smi -> Get-CimInstance -> wmic y autoselecciona checkbox CUDA si hay NVIDIA
; - No descarga InfractiVision-cpu-Win-x64.zip (ya embebido) — evita 404 y 275M duplicados.
; - Si checkbox CUDA marcado (autoseleccionado con NVIDIA), intenta pip install torch==2.6.0+cu124 vía Python del sistema (requiere internet).
; - Modelos 21 MB no se bundlean, se descargan on-demand a %APPDATA%\InfractiVision\models
; Uso: iscc installer/win/online.iss  (requiere dist/InfractiVision/ previo)

#define MyAppName "InfractiVision"
#define MyAppVersion "2.1.0"
#define MyAppPublisher "Abel Moya"
#define MyAppURL "https://github.com/AbelMoyaICSI/InfractiVision"
#define MyRepo "AbelMoyaICSI/InfractiVision"

[Setup]
AppId={{2033F5B2-85DB-456E-9800-9FC2EB030ADB}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={userappdata}\InfractiVision
DefaultGroupName=InfractiVision
AllowNoIcons=yes
OutputDir=..\..\dist
OutputBaseFilename=InfractiVision-Setup-Online
SetupIconFile=..\..\img\icon.ico
UninstallDisplayIcon={app}\InfractiVision.exe
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
DisableDirPage=no
DisableProgramGroupPage=yes
MinVersion=10.0

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Single-file: ONEDIR embebido completo (177M lzma2), instalación offline-capable
Source: "..\..\dist\InfractiVision\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Permissions: users-modify
Source: "..\..\img\icon.ico"; DestDir: "{app}"; Flags: ignoreversion
; VC++ Redist 2015-2022 x64 embebido si existe al compilar (offline-capable, ~24MB extra)
; release.yml lo descarga a installer/win/vc_redist.x64.exe antes de iscc
#ifexist "vc_redist.x64.exe"
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: ignoreversion deleteafterinstall
#endif

[Icons]
Name: "{group}\InfractiVision"; Filename: "{app}\InfractiVision.exe"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"
Name: "{group}\{cm:ProgramOnTheWeb,InfractiVision}"; Filename: "{#MyAppURL}"
Name: "{group}\{cm:UninstallProgram,InfractiVision}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\InfractiVision"; Filename: "{app}\InfractiVision.exe"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\InfractiVision.exe"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,InfractiVision}"; Flags: nowait postinstall skipifsilent

[Code]
var
  DownloadPage: TDownloadWizardPage;
  GpuPage: TWizardPage;
  GpuLabelTitle: TNewStaticText;
  GpuLabelDetail: TNewStaticText;
  GpuLabelVariant: TNewStaticText;
  GpuCudaCheckBox: TNewCheckBox;
  GpuDetected: Boolean;
  GpuName: String;
  GpuDriverOk: Boolean;
  SelectedVariant: String; // 'cuda' o 'cpu' (cpu+pip-cuda cuando checkbox marcado)
  DemoFiles: array of String;
  DemoURLs: array of String;

procedure InitDemoVideos;
begin
  SetArrayLength(DemoFiles, 5);
  DemoFiles[0] := 'Av-Condorcanqui.mp4';
  DemoFiles[1] := 'VID1EDIT ‐ Hecho con Clipchamp.mp4';
  DemoFiles[2] := 'VID2COLISEO.MOV';
  DemoFiles[3] := 'VID2EDIT ‐ Hecho con Clipchamp.mp4';
  DemoFiles[4] := 'VID4EDIT ‐ Hecho con Clipchamp.mp4';
  SetArrayLength(DemoURLs, 5);
  DemoURLs[0] := 'https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/Av-Condorcanqui.mp4?alt=media&token=9ee9bd87-2a0f-4bf2-8acb-445f0bbb48e4';
  DemoURLs[1] := 'https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/VID1EDIT%20%E2%80%90%20Hecho%20con%20Clipchamp.mp4?alt=media&token=b99a2f2d-a765-44bb-a4b4-63c2e8a1357a';
  DemoURLs[2] := 'https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/VID2COLISEO.MOV?alt=media&token=10317415-ed30-4ae1-869f-3c47c31fdaa6';
  DemoURLs[3] := 'https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/VID2EDIT%20%E2%80%90%20Hecho%20con%20Clipchamp.mp4?alt=media&token=9bcae3a5-b76a-4b70-ad5a-ea153cdaec18';
  DemoURLs[4] := 'https://firebasestorage.googleapis.com/v0/b/infractivision-e8c03.firebasestorage.app/o/VID4EDIT%20%E2%80%90%20Hecho%20con%20Clipchamp.mp4?alt=media&token=520a3110-d499-4a9e-b43d-cb054ca48e0a';
end;

function BoolToStrCustom(B: Boolean): String;
begin
  if B then Result := 'True' else Result := 'False';
end;

function NeedsVCRedist(): Boolean;
begin
  Result := not RegKeyExists(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64');
end;

function InstallVCRedist(): Boolean;
var
  VcPath, VcUrl, Cmd: String;
  ResultCode: Integer;
  VcDownloaded: Boolean;
begin
  Result := True; // ya instalado -> éxito
  if not NeedsVCRedist() then
  begin
    Log('InstallVCRedist: VC++ Redist x64 ya instalado, skip');
    Exit;
  end;
  Log('InstallVCRedist: VC++ Redist x64 NO detectado — intentando instalación (requerido para cv2)');
  VcPath := ExpandConstant('{tmp}\vc_redist.x64.exe');
  VcUrl := 'https://aka.ms/vs/17/release/vc_redist.x64.exe';
  VcDownloaded := FileExists(VcPath);

  // Si no estaba embebido, intentar descarga via PowerShell/curl (requiere internet)
  if not VcDownloaded then
  begin
    Log('InstallVCRedist: vc_redist no embebido en {tmp}, intentando descarga desde ' + VcUrl);
    // Método directo: Exec powershell sin captura
    if not FileExists(VcPath) then
    begin
      Cmd := '-NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri ''' + VcUrl + ''' -OutFile ''''' + VcPath + ''''' -UseBasicParsing"';
      if Exec('powershell', Cmd, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      begin
        Log('InstallVCRedist: powershell download ResultCode=' + IntToStr(ResultCode) + ' FileExists=' + BoolToStrCustom(FileExists(VcPath)));
      end;
    end;
    // Fallback: curl si existe (Win10 1803+)
    if not FileExists(VcPath) then
    begin
      Cmd := '-L -o "' + VcPath + '" ' + VcUrl;
      if Exec('curl', Cmd, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
        Log('InstallVCRedist: curl download ResultCode=' + IntToStr(ResultCode));
    end;
    VcDownloaded := FileExists(VcPath);
    if not VcDownloaded then
    begin
      Log('InstallVCRedist: No se pudo obtener vc_redist.x64.exe (sin internet o bloqueado). Se dejará mensaje manual.');
      Result := False;
      Exit;
    end;
  end
  else
    Log('InstallVCRedist: vc_redist embebido encontrado en {tmp}');

  // Instalación silenciosa /quiet /norestart (requiere elevación UAC — el EXE de MS lo pide)
  Log('InstallVCRedist: Ejecutando ' + VcPath + ' /install /quiet /norestart');
  if Exec(VcPath, '/install /quiet /norestart', '', SW_SHOW, ewWaitUntilTerminated, ResultCode) then
  begin
    Log('InstallVCRedist: Exec terminó ResultCode=' + IntToStr(ResultCode));
    // 0 = OK, 1638 = ya instalado, 3010 = requiere reinicio — todos se consideran éxito para cv2
    if (ResultCode = 0) or (ResultCode = 1638) or (ResultCode = 3010) then
    begin
      Log('InstallVCRedist: VC++ instalado correctamente (ResultCode=' + IntToStr(ResultCode) + ')');
      if ResultCode = 3010 then
        Log('InstallVCRedist: Reinicio pendiente para VC++ (3010), cv2 debería funcionar igual sin reiniciar');
      Result := True;
      Exit;
    end
    else
    begin
      Log('InstallVCRedist: VC++ installer falló ResultCode=' + IntToStr(ResultCode) + ' — se mostrará mensaje manual');
      Result := False;
    end;
  end
  else
    Log('InstallVCRedist: Exec falló al lanzar vc_redist');
end;

// ---- Detección GPU NVIDIA dedicada ----

function TryExecAndCapture(const Exe, Params: String; var Output: String): Boolean;
var
  TmpFile: String;
  Cmd: String;
  Lines: TArrayOfString;
  I: Integer;
begin
  TmpFile := ExpandConstant('{tmp}\gpu_detect.txt');
  // Redirige stdout+stderr a archivo
  Cmd := '/C "' + Exe + ' ' + Params + ' > "' + TmpFile + '" 2>&1"';
  if Exec(ExpandConstant('{cmd}'), Cmd, '', SW_HIDE, ewWaitUntilTerminated, I) then
  begin
    if FileExists(TmpFile) then
    begin
      if LoadStringsFromFile(TmpFile, Lines) then
      begin
        Output := '';
        for I := 0 to GetArrayLength(Lines)-1 do
          Output := Output + Lines[I] + #10;
        DeleteFile(TmpFile);
        Result := True;
        Exit;
      end;
      DeleteFile(TmpFile);
    end;
  end;
  Result := False;
end;

function ContainsText(const S, Sub: String): Boolean;
begin
  Result := Pos(Lowercase(Sub), Lowercase(S)) > 0;
end;

function DetectNvidiaGPU(): Boolean;
var
  OutStr: String;
  NvSmiPath: String;
  Tmp: String;
begin
  GpuName := '';
  GpuDriverOk := True;
  Result := False;

  // 1. nvidia-smi en PATH + en Program Files
  if TryExecAndCapture('nvidia-smi', '-L', OutStr) and ContainsText(OutStr, 'GPU') and ContainsText(OutStr, 'NVIDIA') then
  begin
    Result := True;
    // Extraer nombre: primera línea "GPU 0: NVIDIA GeForce RTX 5050 ..."
    Tmp := Trim(OutStr);
    if Pos(':', Tmp) > 0 then
      GpuName := Trim(Copy(Tmp, Pos(':', Tmp)+1, 200))
    else
      GpuName := 'NVIDIA GPU (nvidia-smi)';
    Log('GPU detectada via nvidia-smi -L: ' + GpuName);
    Exit;
  end;

  NvSmiPath := ExpandConstant('{pf}\NVIDIA Corporation\NVSMI\nvidia-smi.exe');
  if FileExists(NvSmiPath) then
  begin
    if TryExecAndCapture('"' + NvSmiPath + '"', '-L', OutStr) and ContainsText(OutStr, 'GPU') then
    begin
      Result := True;
      Tmp := Trim(OutStr);
      if Pos(':', Tmp) > 0 then
        GpuName := Trim(Copy(Tmp, Pos(':', Tmp)+1, 200))
      else
        GpuName := 'NVIDIA GPU';
      Log('GPU detectada via PF nvidia-smi: ' + GpuName);
      Exit;
    end;
  end;

  // 2. PowerShell CIM (moderno, Win10/11) - priorizar sobre wmic deprecado
  if TryExecAndCapture('powershell', '-NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"', OutStr) and ContainsText(OutStr, 'NVIDIA') then
  begin
    Result := True;
    // Tomar primera línea que contenga NVIDIA
    Tmp := Trim(OutStr);
    GpuName := Trim(Tmp);
    // Si hay varias líneas, quedarnos con la que contiene NVIDIA
    if Pos(#10, GpuName) > 0 then
    begin
      // Buscar línea con NVIDIA
      if ContainsText(GpuName, 'NVIDIA') then
      begin
        // Extraer línea NVIDIA (simple: primera no vacía con NVIDIA)
      end;
    end;
    Log('GPU detectada via PowerShell CIM: ' + GpuName);
    Exit;
  end;

  // 3. wmic fallback (deprecado pero aún presente en Win10)
  if TryExecAndCapture('wmic', 'path Win32_VideoController get Name', OutStr) and ContainsText(OutStr, 'NVIDIA') then
  begin
    Result := True;
    Tmp := Trim(OutStr);
    GpuName := Tmp;
    Log('GPU detectada via wmic: ' + GpuName);
    Exit;
  end;

  // 4. powershell legacy WMI
  if TryExecAndCapture('powershell', '-NoProfile -Command "Get-WmiObject Win32_VideoController | Select-Object -ExpandProperty Name"', OutStr) and ContainsText(OutStr, 'NVIDIA') then
  begin
    Result := True;
    GpuName := Trim(OutStr);
    Log('GPU detectada via PowerShell WMI: ' + GpuName);
    Exit;
  end;

  Log('No se detectó GPU NVIDIA dedicada');
  GpuName := '';
  Result := False;
end;

function GetDriverVersionCheck(): Boolean;
var
  OutStr: String;
begin
  // Intento de verificar driver >= 550 (CUDA 12.4). Si no se puede, asumir OK.
  if TryExecAndCapture('nvidia-smi', '--query-gpu=driver_version --format=csv,noheader', OutStr) then
  begin
    Log('Driver version raw: ' + Trim(OutStr));
    // Parse simple: primer número antes de '.'
    // Si falla parse, no bloquear instalación
    Result := True;
    Exit;
  end;
  Result := True;
end;

procedure GpuCudaCheckBoxClick(Sender: TObject);
var
  CheckedStr: String;
begin
  if GpuCudaCheckBox.Checked then
  begin
    SelectedVariant := 'cpu+pip-cuda';
    CheckedStr := 'True';
  end
  else
  begin
    SelectedVariant := 'cpu';
    CheckedStr := 'False';
  end;
  Log('GpuCudaCheckBoxClick: Checked=' + CheckedStr + ' SelectedVariant=' + SelectedVariant);
end;

procedure UpdateGpuPageUI;
begin
  // Opcion 3 sin GCS: base siempre CPU embebida; CUDA via pip autoseleccionado si hay NVIDIA
  if GpuDetected then
  begin
    GpuLabelTitle.Caption := '✅ GPU NVIDIA dedicada detectada';
    if GpuName <> '' then
      GpuLabelDetail.Caption := '   ' + GpuName + ' — ✅ aceleración CUDA autoseleccionada (puedes desmarcar)'
    else
      GpuLabelDetail.Caption := '   GPU detectada — aceleración CUDA autoseleccionada';
    GpuLabelVariant.Caption := '→ Variante base: CPU — 900 MB + CUDA vía pip si está marcado';
    GpuCudaCheckBox.Checked := True; // autoselección cuando detecta NVIDIA
    GpuCudaCheckBox.Enabled := True;
    SelectedVariant := 'cpu+pip-cuda';
    Log('UI GPU: NVIDIA detectada -> CUDA autoseleccionado - ' + GpuName);
  end
  else
  begin
    GpuLabelTitle.Caption := '❌ No se detectó GPU NVIDIA dedicada';
    GpuLabelDetail.Caption := '   Se instalará la variante CPU (compatible con todos los equipos)';
    GpuLabelVariant.Caption := '→ Variante seleccionada: CPU — 900 MB';
    GpuCudaCheckBox.Checked := False;
    GpuCudaCheckBox.Enabled := True; // permitir forzar CUDA manualmente si el usuario quiere
    SelectedVariant := 'cpu';
    Log('UI GPU: CPU seleccionada (sin NVIDIA, checkbox desmarcado)');
  end;
end;

procedure InitializeWizard;
begin
  InitDemoVideos;
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);

  // Página informativa de detección GPU (después de Welcome, antes de SelectDir)
  GpuPage := CreateCustomPage(wpWelcome, 'Detección de hardware', 'InfractiVision detecta automáticamente tu GPU');
  GpuLabelTitle := TNewStaticText.Create(GpuPage);
  GpuLabelTitle.Parent := GpuPage.Surface;
  GpuLabelTitle.Left := 16;
  GpuLabelTitle.Top := 16;
  GpuLabelTitle.Width := 400;
  GpuLabelTitle.Height := 20;
  GpuLabelTitle.Caption := '🔍 Detectando GPU NVIDIA...';
  GpuLabelTitle.Font.Style := [fsBold];
  GpuLabelTitle.Font.Size := 10;

  GpuLabelDetail := TNewStaticText.Create(GpuPage);
  GpuLabelDetail.Parent := GpuPage.Surface;
  GpuLabelDetail.Left := 16;
  GpuLabelDetail.Top := 44;
  GpuLabelDetail.Width := 400;
  GpuLabelDetail.Height := 36;
  GpuLabelDetail.Caption := '   Analizando hardware gráfico. Esto solo toma un segundo...';
  GpuLabelDetail.AutoSize := False;
  GpuLabelDetail.WordWrap := True;

  GpuLabelVariant := TNewStaticText.Create(GpuPage);
  GpuLabelVariant.Parent := GpuPage.Surface;
  GpuLabelVariant.Left := 16;
  GpuLabelVariant.Top := 84;
  GpuLabelVariant.Width := 400;
  GpuLabelVariant.Height := 20;
  GpuLabelVariant.Caption := '';
  GpuLabelVariant.Font.Style := [fsBold];

  GpuCudaCheckBox := TNewCheckBox.Create(GpuPage);
  GpuCudaCheckBox.Parent := GpuPage.Surface;
  GpuCudaCheckBox.Left := 16;
  GpuCudaCheckBox.Top := 112;
  GpuCudaCheckBox.Width := 400;
  GpuCudaCheckBox.Height := 17;
  GpuCudaCheckBox.Caption := 'Instalar aceleración CUDA (requiere Python 3.10 + internet, ~2 GB)';
  GpuCudaCheckBox.Checked := False;
  GpuCudaCheckBox.Enabled := True;
  GpuCudaCheckBox.OnClick := @GpuCudaCheckBoxClick;

  // Valores iniciales (se actualizan en CurPageChanged)
  GpuDetected := False;
  GpuName := '';
  SelectedVariant := 'cpu';
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = GpuPage.ID then
  begin
    WizardForm.NextButton.Enabled := False;
    GpuLabelTitle.Caption := '🔍 Detectando GPU NVIDIA...';
    GpuLabelDetail.Caption := '   Analizando hardware gráfico...';
    GpuLabelVariant.Caption := '';
    // Forzar repintado
    GpuPage.Surface.Repaint;

    // Detección síncrona (rápida, <1s). Si tarda, Next sigue bloqueado.
    GpuDetected := DetectNvidiaGPU();
    if GpuDetected then
      GpuDriverOk := GetDriverVersionCheck()
    else
      GpuDriverOk := True;
    UpdateGpuPageUI;
    WizardForm.NextButton.Enabled := True;
    // Auto-continuar solo: no requiere clic si el usuario no quiere esperar
    // Nota: dejamos al usuario ver el resultado; él hace Next. Si quieres auto-skip,
    // descomenta la línea siguiente para avanzar automáticamente tras 1.5s:
    // WizardForm.NextButton.OnClick(WizardForm.NextButton);
  end;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  // No saltar página GPU
  Result := False;
end;

// ---- Pip CUDA on-demand (opcion 3 single-file: stub CPU + pip si hay NVIDIA) ----

function FindSystemPython(var PythonExe: String): Boolean;
var
  OutStr, Candidate, RegPath: String;
begin
  Result := False;
  PythonExe := '';
  // 1. py launcher
  if TryExecAndCapture('py', '-3 --version', OutStr) and ContainsText(OutStr, 'Python') then
  begin
    PythonExe := 'py';
    Log('FindSystemPython: py -3 -> ' + Trim(OutStr));
    Result := True;
    Exit;
  end;
  // 2. python
  if TryExecAndCapture('python', '--version', OutStr) and ContainsText(OutStr, 'Python') then
  begin
    PythonExe := 'python';
    Log('FindSystemPython: python -> ' + Trim(OutStr));
    Result := True;
    Exit;
  end;
  // 3. python3
  if TryExecAndCapture('python3', '--version', OutStr) and ContainsText(OutStr, 'Python') then
  begin
    PythonExe := 'python3';
    Log('FindSystemPython: python3 -> ' + Trim(OutStr));
    Result := True;
    Exit;
  end;
  // 4. Registry fallback
  if RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.10\InstallPath', '', Candidate) and FileExists(Candidate + '\python.exe') then
  begin
    PythonExe := Candidate + '\python.exe';
    Log('FindSystemPython: HKLM 3.10 -> ' + PythonExe);
    Result := True;
    Exit;
  end;
  if RegQueryStringValue(HKCU, 'SOFTWARE\Python\PythonCore\3.10\InstallPath', '', Candidate) and FileExists(Candidate + '\python.exe') then
  begin
    PythonExe := Candidate + '\python.exe';
    Log('FindSystemPython: HKCU 3.10 -> ' + PythonExe);
    Result := True;
    Exit;
  end;
  Log('FindSystemPython: no Python del sistema encontrado');
end;

function TryPipInstallCuda(AppPath: String): Boolean;
var
  PythonExe, PipArgs, PipLog: String;
  ResultCode: Integer;
begin
  Result := False;
  // Respeta autoselección del checkbox: si está desmarcado, no instalar CUDA
  if not GpuCudaCheckBox.Checked then
  begin
    Log('TryPipInstallCuda: checkbox desmarcado, skip pip (SelectedVariant=' + SelectedVariant + ')');
    Exit;
  end;
  if not GpuDetected then
    Log('TryPipInstallCuda: checkbox marcado sin GPU detectada (forzado manual), intentando pip igual');
  if not FindSystemPython(PythonExe) then
  begin
    Log('TryPipInstallCuda: sin Python del sistema, se queda CPU (app fallback torch.cuda.is_available)');
    SuppressibleMsgBox('No se encontró Python del sistema para instalar aceleración CUDA.' + #13#10 + 'La app funcionará en modo CPU. Instala Python 3.10 y re-ejecuta el instalador para CUDA.', mbInformation, MB_OK, IDOK);
    Exit;
  end;
  Log('TryPipInstallCuda: Python=' + PythonExe + ' AppPath=' + AppPath);
  PipLog := ExpandConstant('{tmp}\pip_cuda.log');
  // Upgrade pip silencioso
  PipArgs := '-3 -m pip install --upgrade pip --disable-pip-version-check > "' + PipLog + '" 2>&1';
  if Pos('py', PythonExe) = 1 then
    PipArgs := '-3 -m pip install --upgrade pip --disable-pip-version-check > "' + PipLog + '" 2>&1'
  else
    PipArgs := '-m pip install --upgrade pip --disable-pip-version-check > "' + PipLog + '" 2>&1';
  Log('Pip upgrade: ' + PythonExe + ' ' + PipArgs);
  Exec(ExpandConstant('{cmd}'), '/C "' + PythonExe + ' ' + PipArgs + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // Instalar torch CUDA sobre el embebido: --target {app}\_internal para que el bootloader lo vea
  // Usa --no-warn-script-location y --disable-pip-version-check para evitar prompts
  if PythonExe = 'py' then
    PipArgs := '-3 -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 --target "' + AppPath + '\_internal" --no-warn-script-location --disable-pip-version-check --no-input >> "' + PipLog + '" 2>&1'
  else
    PipArgs := '-m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 --target "' + AppPath + '\_internal" --no-warn-script-location --disable-pip-version-check --no-input >> "' + PipLog + '" 2>&1';
  Log('Pip CUDA: ' + PythonExe + ' ' + PipArgs);
  if Exec(ExpandConstant('{cmd}'), '/C "' + PythonExe + ' ' + PipArgs + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
    begin
      Log('TryPipInstallCuda OK resultCode=0 log=' + PipLog);
      Result := True;
    end
    else
    begin
      Log('TryPipInstallCuda fallo resultCode=' + IntToStr(ResultCode) + ' log=' + PipLog + ' — queda CPU');
      SuppressibleMsgBox('No se pudo instalar CUDA vía pip (código ' + IntToStr(ResultCode) + ').' + #13#10 + 'La app funcionará en CPU. Revisa ' + PipLog, mbInformation, MB_OK, IDOK);
    end;
  end
  else
    Log('TryPipInstallCuda: Exec fallo');
end;

procedure DownloadDemoVideos(AppPath: String);
var
  I, Added: Integer;
  VDir, TmpFile, DestFile: String;
begin
  VDir := AppPath + '\videos';
  if not ForceDirectories(VDir) then
    Log('No se pudo crear ' + VDir);
  DownloadPage.Clear;
  Added := 0;
  for I := 0 to GetArrayLength(DemoFiles)-1 do
  begin
    DestFile := VDir + '\' + DemoFiles[I];
    if FileExists(DestFile) then
    begin
      Log('Video ya existe, skip asset: ' + DemoFiles[I]);
      Continue;
    end;
    Log('Queue asset: ' + DemoFiles[I]);
    DownloadPage.Add(DemoURLs[I], DemoFiles[I], '');
    Inc(Added);
  end;
  if Added = 0 then
  begin
    Log('Todos los videos demo ya existen — no hay assets que descargar');
    Exit;
  end;
  DownloadPage.Show;
  try
    DownloadPage.Download;
  except
    if DownloadPage.AbortedByUser then
      Log('Descarga de videos abortada por usuario')
    else
      SuppressibleMsgBox('Error descargando videos: ' + GetExceptionMessage + #13#10 + 'La app los reintentará al primer inicio.', mbError, MB_OK, IDOK);
    Exit;
  finally
    DownloadPage.Hide;
  end;
  for I := 0 to GetArrayLength(DemoFiles)-1 do
  begin
    TmpFile := ExpandConstant('{tmp}\' + DemoFiles[I]);
    DestFile := VDir + '\' + DemoFiles[I];
    if FileExists(TmpFile) then
    begin
      if not FileCopy(TmpFile, DestFile, False) then
        Log('Fallo moviendo asset ' + DemoFiles[I] + ' de {tmp} a videos/')
      else
        Log('Asset listo: ' + DestFile);
      DeleteFile(TmpFile);
    end;
  end;
end;

procedure EnsureModelsPreFetched(AppPath: String);
var
  MDir: String;
begin
  MDir := AppPath + '\models';
  if not ForceDirectories(MDir) then
    Log('No se pudo crear ' + MDir);
  Log('Modelos se descargaran on-demand al primer arranque a %APPDATA%\InfractiVision\models');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  VcOk: Boolean;
begin
  if CurStep = ssPostInstall then
  begin
    // Single-file: ONEDIR ya embebido en {app}, no descarga zip redundante (evita 404 + 275M)
    Log('Single-file: ONEDIR ya en {app}, salto descarga zip');
    // 0. VC++ Redist x64 — requerido para cv2 (DLL load failed si falta). Auto-instala si no está.
    VcOk := True;
    if NeedsVCRedist() then
    begin
      Log('CurStepChanged: VC++ faltante detectado, lanzando InstallVCRedist');
      VcOk := InstallVCRedist();
      if not VcOk and NeedsVCRedist() then
        SuppressibleMsgBox('No se pudo instalar Microsoft Visual C++ 2015-2022 Redistributable (x64) automáticamente.' + #13#10 + 'La app fallará con "DLL load failed while importing cv2".' + #13#10 + 'Instálalo manualmente desde https://aka.ms/vs/17/release/vc_redist.x64.exe y reinicia.', mbInformation, MB_OK, IDOK);
    end;
    // 1. Si hay GPU, intenta pip CUDA sobre el embebido
    TryPipInstallCuda(ExpandConstant('{app}'));
    // 2. Videos demo
    DownloadDemoVideos(ExpandConstant('{app}'));
    EnsureModelsPreFetched(ExpandConstant('{app}'));
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then begin end;
end;
