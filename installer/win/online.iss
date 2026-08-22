; InfractiVision ONLINE Installer - Inno Setup 6
; Stub 8MB: detecta GPU y descarga artefacto correcto desde GitHub Releases
; Uso: iscc installer/win/online.iss

#define MyAppName "InfractiVision"
#define MyAppVersion "2.1.0"
#define MyAppPublisher "Abel Moya"
#define MyAppURL "https://github.com/AbelMoyaICSI/InfractiVision"
#define RepoBase "https://github.com/AbelMoyaICSI/InfractiVision/releases/latest/download"

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
    Result := '{#RepoBase}/InfractiVision-cuda-Win-x64.zip'
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
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), nil);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  URL, ZipPath, ExePath: String;
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
    // Usa ExpandConstant + ShellExec con powershell Expand-Archive
    if not Exec('powershell.exe', '-NoProfile -Command "Expand-Archive -Force ''' + ZipPath + ''' ''' + ExePath + '''"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then begin
      MsgBox('Fallo extrayendo artefacto', mbError, MB_OK);
      Result := False;
    end else if ResultCode <> 0 then begin
      MsgBox('Descompresion fallo (codigo ' + IntToStr(ResultCode) + '). Intenta de nuevo.', mbError, MB_OK);
      Result := False;
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
