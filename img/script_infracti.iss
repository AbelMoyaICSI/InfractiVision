; --- Inno Setup Script (per-user, Inno 5.x) ---

[Setup]
AppId={{2033F5B2-85DB-456E-9800-9FC2EB030ADB}
AppName=InfractiVision
AppVersion=1.0
AppVerName=InfractiVision 1.0
AppPublisher=Abel Moya
AppPublisherURL=https://github.com/AbelMoyaICSI/InfractiVision
AppSupportURL=https://github.com/AbelMoyaICSI/InfractiVision
AppUpdatesURL=https://github.com/AbelMoyaICSI/InfractiVision

; Instala en AppData del usuario (evita permisos de admin)
DefaultDirName={userappdata}\InfractiVision
DefaultGroupName=InfractiVision
AllowNoIcons=yes

; Dónde se guarda el instalador generado
OutputDir=C:\Users\Abel\Desktop
OutputBaseFilename=Setup

; Icono del instalador / desinstalador
SetupIconFile=C:\Users\Abel\Desktop\InfractiVision\img\icon.ico
UninstallDisplayIcon={app}\InfractiVision.exe

; Compresión
Compression=lzma
SolidCompression=yes

; En Inno 5.x no existe "lowest": usar "none" para no pedir elevación
PrivilegesRequired=none

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Copia el ejecutable único generado por PyInstaller
Source: "C:\Users\Abel\Desktop\InfractiVision\dist\InfractiVision.exe"; DestDir: "{app}"; Flags: ignoreversion
; Copia el icono para los accesos directos
Source: "C:\Users\Abel\Desktop\InfractiVision\img\icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Acceso directo en grupo de inicio
Name: "{group}\InfractiVision"; Filename: "{app}\InfractiVision.exe"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"
; Web
Name: "{group}\{cm:ProgramOnTheWeb,InfractiVision}"; Filename: "https://github.com/AbelMoyaICSI/InfractiVision"
; Desinstalar
Name: "{group}\{cm:UninstallProgram,InfractiVision}"; Filename: "{uninstallexe}"
; Acceso directo en escritorio (opcional)
Name: "{commondesktop}\InfractiVision"; Filename: "{app}\InfractiVision.exe"; WorkingDir: "{app}"; Tasks: desktopicon; IconFilename: "{app}\icon.ico"

[Run]
; Lanza la app al finalizar usando el directorio correcto
Filename: "{app}\InfractiVision.exe"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,InfractiVision}"; Flags: nowait postinstall skipifsilent

