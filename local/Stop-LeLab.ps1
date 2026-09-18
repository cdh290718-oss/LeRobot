$ErrorActionPreference = 'Stop'
$pidFile = "$PSScriptRoot\logs\server.pid"
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Host 'No background server PID file. For a visible launcher, press Ctrl+C in its window.'; exit 0 }
$serverProcessId = [int](Get-Content -LiteralPath $pidFile)
$rootProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$serverProcessId"
if (-not $rootProcess) { Write-Host 'Background launcher is not running.'; exit 0 }
if ($rootProcess.ExecutablePath -ne 'G:\LeRobot\.venv\Scripts\python.exe' -or $rootProcess.CommandLine -notmatch 'lelab.scripts.lelab') { throw 'PID does not match this LeLab launcher; nothing stopped.' }
$childProcesses = Get-CimInstance Win32_Process -Filter "ParentProcessId=$serverProcessId"
foreach ($child in $childProcesses) {
    if ($child.ExecutablePath -in @('G:\LeRobot\.venv\Scripts\python.exe', 'G:\LeRobot\python\cpython-3.12-windows-x86_64-none\python.exe') -and $child.CommandLine -match 'lelab.scripts.lelab') { Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue }
}
Stop-Process -Id $serverProcessId -ErrorAction SilentlyContinue
Write-Host 'LeLab background server stopped.'

