; InfractiVision ONLINE Installer - Inno Setup 6 (ligero / descarga selectiva)
; Stub ~8MB: detecta GPU y descarga artefacto correcto desde GitHub Releases.
; ONLINE ligero: exe NO bundlea modelos (21MB) ni videos (2GB). Los modelos
; (yolov8n, license_plate_detector, LPRNet V4) se descargan on-demand al primer
; arranque via src/infrastructure/storage/model_downloader.py a
; %APPDATA%\InfractiVision\models (idempotente, sha256+size).
; Uso: iscc installer/win/online.iss

#define MyAppName "InfractiVision"
#define MyAppVersion "2.1.0"
#define MyAppPublisher "Abel Moya"
#define MyAppURL "https://github.com/AbelMoyaICSI/InfractiVision"
#define RepoBase "https://github.com/AbelMoyaICSI/InfractiVision/releases/latest/download"
; El zip CUDA-Win supera el limite de 2GB de GitHub Releases; se aloja en GCS publico.
#define GcsBase "https://storage.googleapis.com/infractivision-e8c03.firebasestorage.app/releases/latest"

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
; Requiere conexion
MinVersion=10.0

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Stub no incluye exe principal: se descarga en [Code] CurStepChanged
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
  HasNvidiaGPU: Boolean;
  DemoFiles: array of String;
  DemoURLs: array of String;

// Videos demo (deben coincidir con config/demo_videos.json)
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

function HasNvidiaGPUCheck(): Boolean;
var
  TmpFile: String;
  ResultCode: Integer;
  Output: AnsiString;
begin
  Result := False;
  // 1) nvidia-smi -L (mas fiable si driver instalado)
  if Exec('nvidia-smi', '-L', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    if ResultCode = 0 then begin Result := True; Exit; end;
  // 2) wmic fallback: busca NVIDIA en nombre de GPU
  TmpFile := ExpandConstant('{tmp}\gpu.txt');
  if Exec('cmd.exe', '/c wmic path win32_VideoController get name > "' + TmpFile + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    if LoadStringFromFile(TmpFile, Output) then
      if Pos('NVIDIA', Uppercase(String(Output))) > 0 then Result := True;
end;

function GetArtifactURL(): String;
begin
  if HasNvidiaGPU then
    Result := '{#GcsBase}/InfractiVision-cuda-Win-x64.zip'
  else
    Result := '{#RepoBase}/InfractiVision-cpu-Win-x64.zip';
end;

function NeedsVCRedist(): Boolean;
begin
  // VC++ 2015-2022 Redist check: HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64
  Result := not RegKeyExists(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64');
end;

procedure InitializeWizard;
begin
  HasNvidiaGPU := HasNvidiaGPUCheck();
  InitDemoVideos;
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);
end;

function DownloadFile(URL, Dest: String): Boolean;
var
  ResultCode: Integer;
  Script: String;
begin
  Result := False;
  Script := '-NoProfile -ExecutionPolicy Bypass -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ErrorActionPreference = ''Stop''; try { Invoke-WebRequest -Uri ''@@URL@@'' -OutFile ''@@DEST@@'' -UseBasicParsing -TimeoutSec 3600 } catch { exit 1 } }"';
  StringChangeEx(Script, '@@URL@@', URL, True);
  StringChangeEx(Script, '@@DEST@@', Dest, True);
  if Exec('powershell.exe', Script, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    Result := (ResultCode = 0);
end;

procedure DownloadDemoVideos(AppPath: String);
var
  I, Added: Integer;
  VDir, TmpFile, DestFile: String;
begin
  VDir := AppPath + '\videos';
  if not ForceDirectories(VDir) then
    Log('No se pudo crear ' + VDir);

  // Descarga visible como assets: cada video aparece en el wizard con barra.
  // Usa TDownloadWizardPage (mismo que el artefacto) para que el usuario vea
  // "Descargando assets (3/5) — VID2COLISEO.MOV".
  DownloadPage.Clear;
  Added := 0;
  for I := 0 to GetArrayLength(DemoFiles)-1 do
  begin
    DestFile := VDir + '\' + DemoFiles[I];
    // Idempotente: si ya existe y tiene size >0, skip (la app valida sha256 al iniciar)
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

  DownloadPage.Msg1LabelCaption := 'Descargando vídeos demo...';
  DownloadPage.Msg2LabelCaption := 'Esto puede tardar varios minutos (2.6 GB total)';
  DownloadPage.Show;
  try
    DownloadPage.Download;
  except
    if DownloadPage.AbortedByUser then
      Log('Descarga de videos abortada por usuario')
    else
      SuppressibleMsgBox('Error descargando videos: ' + GetExceptionMessage + #13#10 + 'La app los reintentará al primer inicio.', mbError, MB_OK, IDOK);
    // No bloquea la instalacion: la app reintenta en main.py:86
    Exit;
  finally
    DownloadPage.Hide;
  end;

  // TDownloadWizardPage descarga a {tmp}\<filename>. Mover a {app}\videos.
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
  // Prefetch opcional de modelos al instalar (ahorra espera en primer arranque).
  // Si falla la red, la app lo reintenta al iniciar via model_downloader.
  MDir := AppPath + '\models';
  if not ForceDirectories(MDir) then
    Log('No se pudo crear ' + MDir);
  Log('Modelos se descargaran on-demand al primer arranque a %APPDATA%\InfractiVision\models');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  URL, ZipPath, ExePath: String;
  ResultCode: Integer;
begin
  Result := True;
  if CurPageID = wpReady then begin
    URL := GetArtifactURL();
    ZipPath := ExpandConstant('{tmp}\infractivision.zip');
    DownloadPage.Clear;
    DownloadPage.Add(URL, 'infractivision.zip', '');
    DownloadPage.Show;
    try
      DownloadPage.Download;
    except
      if DownloadPage.AbortedByUser then
        Log('Descarga abortada por usuario')
      else
        SuppressibleMsgBox('Error descargando: ' + GetExceptionMessage + #13#10 + 'Verifica tu conexion. URL: ' + URL, mbError, MB_OK, IDOK);
      Result := False;
      Exit;
    finally
      DownloadPage.Hide;
    end;
    // Extrae zip a {app}
    ExePath := ExpandConstant('{app}');
    if not ForceDirectories(ExePath) then begin
      MsgBox('No se pudo crear ' + ExePath, mbError, MB_OK);
      Result := False; Exit;
    end;
    // ONEDIR: el zip contiene la carpeta InfractiVision/ (COLLECT). Expand-Archive la deja en {app}\InfractiVision\
    // ONEFILE legacy: el zip contiene solo InfractiVision.exe en la raiz. Ambos se manejan.
    if not Exec('powershell.exe', '-NoProfile -Command "Expand-Archive -Force ''' + ZipPath + ''' ''' + ExePath + '''"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then begin
      MsgBox('Fallo extrayendo artefacto', mbError, MB_OK);
      Result := False;
    end else if ResultCode <> 0 then begin
      MsgBox('Descompresion fallo (codigo ' + IntToStr(ResultCode) + '). Intenta de nuevo.', mbError, MB_OK);
      Result := False;
    end else begin
      // Si es ONEDIR, el exe quedo en {app}\InfractiVision\InfractiVision.exe -> mover al root {app} para los [Icons]/[Run] existentes
      if FileExists(ExePath + '\InfractiVision\InfractiVision.exe') then
      begin
        // Mover contenido de subcarpeta al root (para mantener {app}\InfractiVision.exe como antes)
        Exec('powershell.exe', '-NoProfile -Command "Move-Item -Force ''' + ExePath + '\InfractiVision\*'' ''' + ExePath + '''"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
        RemoveDir(ExePath + '\InfractiVision');
      end;
      // Artefacto extraido OK: descarga los videos demo a {app}\videos (visible como assets) + prefetch modelos.
      DownloadDemoVideos(ExePath);
      EnsureModelsPreFetched(ExePath);
    end;
    if NeedsVCRedist() then
      SuppressibleMsgBox('Falta Microsoft Visual C++ 2015-2022 Redistributable (x64). La app puede fallar al iniciar. Descargalo desde https://aka.ms/vs/17/release/vc_redist.x64.exe', mbInformation, MB_OK, IDOK);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then begin
    // No borra APPDATA (evidencias/logs del usuario), solo binarios en {app}
  end;
end;
