# Install AWS Session Manager plugin on Windows (required for ssm start-session).
# Does not buy compute or start sessions.
$ErrorActionPreference = "Stop"
$url = "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/windows/SessionManagerPlugin.msi"
$msi = Join-Path $env:TEMP "SessionManagerPlugin.msi"
Write-Host "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $msi
Write-Host "Installing (may prompt for elevation)…"
Start-Process msiexec.exe -ArgumentList "/i `"$msi`" /quiet" -Wait -Verb RunAs
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
Write-Host "Verify with: session-manager-plugin"
Get-Command session-manager-plugin -ErrorAction SilentlyContinue | Format-List
