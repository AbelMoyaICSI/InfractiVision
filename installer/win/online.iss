; InfractiVision Setup Online - Prueba Windows (offline embebido, sin versionado)
; ONEDIR embebido: no descarga cpu/cuda zip, solo vídeos demo como assets visibles.
; Modelos 21 MB no se bundlean, se descargan on-demand al primer arranque a
; %APPDATA%\InfractiVision\models via model_downloader.py (GitHub Releases luego).
; Uso prueba: iscc installer/win/online.iss  (requiere dist/InfractiVision/ previo)

#define MyAppName "InfractiVision"
#define MyAppVersion "2.1.0"
#define MyAppPublisher "Abel Moya"
#define MyAppURL "https://github.com/AbelMoyaICSI/InfractiVision"

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
; Offline embebido (prueba): ONEDIR completo generado por PyInstaller
Source: "..\..\dist\InfractiVision\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
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

procedure InitializeWizard;
begin
  InitDemoVideos;
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);
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
