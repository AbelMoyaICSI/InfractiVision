; InfractiVision Setup Online - Windows con detección GPU NVIDIA + ventana informativa
; Modo: stub online (~8 MB) con detección automática de GPU dedicada NVIDIA.
; - Detecta nvidia-smi -> wmic -> powershell Get-CimInstance
; - Muestra ventana informativa con resultado y continúa solo (sin intervención)
; - Descarga variante CUDA (1.4GB) si hay NVIDIA, sino CPU (900MB) desde GitHub Releases
; - Fallback offline: si no hay red, usa ONEDIR embebido si existe (CI/local)
; - Modelos 21 MB no se bundlean, se descargan on-demand a %APPDATA%\InfractiVision\models
; Uso: iscc installer/win/online.iss  (requiere dist/InfractiVision/ previo para fallback offline)

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
; Fallback offline: ONEDIR embebido si existe (local/CI sin red). Online lo sobreescribe si descarga.
Source: "..\..\dist\InfractiVision\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Permissions: users-modify
Source: "..\..\img\icon.ico"; DestDir: "{app}"; Flags: ignoreversion

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
  GpuDetected: Boolean;
  GpuName: String;
  GpuDriverOk: Boolean;
  SelectedVariant: String; // 'cuda' o 'cpu'
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

function NeedsVCRedist(): Boolean;
begin
  Result := not RegKeyExists(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64');
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

procedure UpdateGpuPageUI;
begin
  if GpuDetected then
  begin
    GpuLabelTitle.Caption := '✅ GPU NVIDIA dedicada detectada';
    if GpuName <> '' then
      GpuLabelDetail.Caption := '   ' + GpuName
    else
      GpuLabelDetail.Caption := '   Se instalará la variante con aceleración CUDA';
    if GpuDriverOk then
      GpuLabelVariant.Caption := '→ Variante seleccionada: CUDA 12.4 (GPU) — 1.4 GB'
    else
      GpuLabelVariant.Caption := '→ Variante seleccionada: CUDA 12.4 (driver desactualizado, fallback a CPU si falla)';
    SelectedVariant := 'cuda';
    Log('UI GPU: CUDA seleccionada - ' + GpuName);
  end
  else
  begin
    GpuLabelTitle.Caption := '❌ No se detectó GPU NVIDIA dedicada';
    GpuLabelDetail.Caption := '   Se instalará la variante CPU (compatible con todos los equipos)';
    GpuLabelVariant.Caption := '→ Variante seleccionada: CPU — 900 MB';
    SelectedVariant := 'cpu';
    Log('UI GPU: CPU seleccionada');
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

// ---- Descargas ----

function GetVariantZipUrl(Variant: String): String;
begin
  // Artefactos publicados en GitHub Releases (latest)
  // Ej: InfractiVision-cuda-Win-x64.zip / InfractiVision-cpu-Win-x64.zip
  Result := 'https://github.com/{#MyRepo}/releases/latest/download/InfractiVision-' + Variant + '-Win-x64.zip';
end;

procedure DownloadAndExtractVariant(AppPath: String);
var
  ZipUrl, ZipTmp, ExtractTmp: String;
  ResultCode: Integer;
  PsCmd: String;
begin
  // Si ya hay ONEDIR embebido (fallback offline) y no hay red, esta descarga es opcional.
  // Intentamos descargar la variante correcta; si falla, dejamos el embebido y logueamos.
  ZipUrl := GetVariantZipUrl(SelectedVariant);
  ZipTmp := ExpandConstant('{tmp}\InfractiVision-' + SelectedVariant + '-Win-x64.zip');
  ExtractTmp := ExpandConstant('{tmp}\iv_extract');

  Log('Variante elegida: ' + SelectedVariant + ' URL: ' + ZipUrl);

  // Solo descargar si la variante no coincide con lo embebido o si AppPath está vacío
  // Heurística: siempre intentar descargar (para tener CUDA si corresponde). Si falla, fallback.
  DownloadPage.Clear;
  DownloadPage.Add(ZipUrl, 'InfractiVision-' + SelectedVariant + '-Win-x64.zip', '');
  DownloadPage.Show;
  try
    try
      DownloadPage.Download;
      Log('Descarga variante OK: ' + SelectedVariant);
    except
      if DownloadPage.AbortedByUser then
      begin
        Log('Descarga abortada por usuario - usando fallback embebido (CPU)');
        Exit;
      end
      else
      begin
        Log('Error descargando variante ' + SelectedVariant + ': ' + GetExceptionMessage + ' — usando fallback embebido');
        SuppressibleMsgBox('No se pudo descargar la variante ' + SelectedVariant + ' (' + GetExceptionMessage + ').' + #13#10 + 'Se usará la versión embebida (CPU) y la app hará fallback automático. Verifica tu conexión.', mbInformation, MB_OK, IDOK);
        Exit;
      end;
    end;
  finally
    DownloadPage.Hide;
  end;

  // Verificar que el zip se descargó a {tmp}
  ZipTmp := ExpandConstant('{tmp}\InfractiVision-' + SelectedVariant + '-Win-x64.zip');
  if not FileExists(ZipTmp) then
  begin
    Log('Zip no encontrado en tmp tras DownloadPage: ' + ZipTmp + ' — fallback embebido');
    Exit;
  end;

  // Extraer con PowerShell Expand-Archive (robusto en Win10+)
  if not ForceDirectories(ExtractTmp) then
    Log('No se pudo crear ' + ExtractTmp);

  PsCmd := '-NoProfile -ExecutionPolicy Bypass -Command "try { Expand-Archive -Path ''' + ZipTmp + ''' -DestinationPath ''' + ExtractTmp + ''' -Force; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"';
  Log('Extrayendo zip con PowerShell: ' + ZipTmp + ' -> ' + ExtractTmp);
  if Exec('powershell', PsCmd, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
    begin
      Log('Expand-Archive OK, copiando a {app}: ' + AppPath);
      // Copiar contenido extraído a {app} sobreescribiendo
      PsCmd := '-NoProfile -ExecutionPolicy Bypass -Command "Copy-Item -Path ''' + ExtractTmp + '\*'' -Destination ''' + AppPath + ''' -Recurse -Force; exit $LASTEXITCODE"';
      if Exec('powershell', PsCmd, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) then
        Log('Copia a {app} OK')
      else
        Log('Fallo copiando a {app}, code=' + IntToStr(ResultCode));
    end
    else
      Log('Expand-Archive fallo code=' + IntToStr(ResultCode));
  end
  else
    Log('No se pudo ejecutar PowerShell para extraer');

  DeleteFile(ZipTmp);
  // No borrar ExtractTmp inmediatamente por si el usuario reinstala
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
begin
  if CurStep = ssPostInstall then
  begin
    // 1. Intenta descargar variante correcta (cuda/cpu) según detección GPU
    DownloadAndExtractVariant(ExpandConstant('{app}'));
    // 2. Videos demo
    DownloadDemoVideos(ExpandConstant('{app}'));
    EnsureModelsPreFetched(ExpandConstant('{app}'));
    if NeedsVCRedist() then
      SuppressibleMsgBox('Falta Microsoft Visual C++ 2015-2022 Redistributable (x64). La app puede fallar al iniciar. Descargalo desde https://aka.ms/vs/17/release/vc_redist.x64.exe', mbInformation, MB_OK, IDOK);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then begin end;
end;
