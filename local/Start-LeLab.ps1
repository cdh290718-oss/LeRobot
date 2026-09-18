param([switch]$NoOpen)
. "$PSScriptRoot\environment.ps1"
try {
    $health = Invoke-RestMethod 'http://127.0.0.1:8000/health' -TimeoutSec 2
    if ($health.status -eq 'ok' -and $health.message -eq 'FastAPI server is running') {
        if (-not $NoOpen) { Start-Process 'http://127.0.0.1:8000' }
        exit 0
    }
} catch {}
if ($NoOpen) { & 'G:\LeRobot\.venv\Scripts\lelab.exe' --no-open }
else { & 'G:\LeRobot\.venv\Scripts\lelab.exe' }
exit $LASTEXITCODE
