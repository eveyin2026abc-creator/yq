param(
    [string]$TaskName = "msmodeling Nightly Board",
    [string]$Bind = "0.0.0.0",
    [int]$Port = 8767,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task: $TaskName"
    exit 0
}

$PublicDir = Join-Path $PSScriptRoot "public"
if (-not (Test-Path (Join-Path $PublicDir "index.html"))) {
    throw "public/index.html was not found."
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    throw "Python 3 was not found in PATH."
}

$Arguments = "-m http.server $Port --bind $Bind --directory `"$PublicDir`""
$Action = New-ScheduledTaskAction -Execute $Python.Source -Argument $Arguments -WorkingDirectory $PSScriptRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Serve the msmodeling Nightly Board on the trusted LAN." -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Installed and started: $TaskName"
Write-Host "Listening on ${Bind}:$Port"
