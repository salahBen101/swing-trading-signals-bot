# Register a post-close RSI(2) scanner task.
#
# The research model confirms a daily signal at the close and fills the next session's open.
# A 15:30 or 15:55 run is useful only as a manual, provisional observation and must not send
# an alert that looks tradeable. The scanner itself therefore suppresses pre-close alerts; this
# scheduler defaults to 16:10 ET so the daily-feed close has time to settle before it sends a
# confirmed paper-trading alert for the next
# open. It places no orders.
#
# Usage (run once from this folder):
#   powershell -ExecutionPolicy Bypass -File .\setup_daily_scan.ps1
#   powershell -ExecutionPolicy Bypass -File .\setup_daily_scan.ps1 -Times 16:10
#   powershell -ExecutionPolicy Bypass -File .\setup_daily_scan.ps1 -Remove

param(
    [string[]]$Times = @("16:10"),
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$projectDir = $PSScriptRoot
$python     = Join-Path $projectDir ".venv\Scripts\python.exe"
$script     = Join-Path $projectDir "scanner\rsi2_scanner.py"

if (-not (Test-Path $python)) { throw "Python not found in .venv" }
if (-not (Test-Path $script)) { throw "Scanner not found at $script" }

# Scheduled Tasks fire in local time, so convert the requested ET time for this machine.
$localNow = Get-Date
$etNow = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId($localNow.ToUniversalTime(), 'Eastern Standard Time')
$offsetHours = [math]::Round(($localNow - $etNow).TotalHours)

Write-Host ""
Write-Host "Machine local time : $($localNow.ToString('HH:mm'))"
Write-Host "Equivalent ET      : $($etNow.ToString('HH:mm'))"
Write-Host "Offset applied     : $offsetHours hour(s)"
Write-Host ""

foreach ($t in $Times) {
    $label    = $t.Replace(":", "")
    $taskName = "RSI2 Scanner $label"

    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "Removed existing '$taskName'"
    }
    if ($Remove) { continue }

    $runTime = (Get-Date $t).AddHours($offsetHours)
    $action = New-ScheduledTaskAction `
        -Execute $python `
        -Argument "`"$script`" --no-options" `
        -WorkingDirectory $projectDir

    $trigger = New-ScheduledTaskTrigger -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
        -At $runTime

    $settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -DontStopIfGoingOnBatteries `
        -AllowStartIfOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "RSI(2) confirmed post-close scan at $t ET. Paper-trading alerts only; no orders." | Out-Null

    Write-Host ("Registered '{0}' -> runs {1} local (= {2} ET)" -f `
        $taskName, $runTime.ToString('HH:mm'), $t) -ForegroundColor Green
}

if ($Remove) {
    Write-Host ""
    Write-Host "All scanner tasks removed." -ForegroundColor Yellow
    return
}

Write-Host ""
Write-Host "The task runs on weekdays, including market holidays. The scanner suppresses stale-data"
Write-Host "alerts; review Task Scheduler history if it reports an upstream data failure."
Write-Host ""
Write-Host "Test one right now:"
Write-Host "    Start-ScheduledTask -TaskName 'RSI2 Scanner 1610'"
Write-Host ""
Write-Host "Remove it later:"
Write-Host "    powershell -ExecutionPolicy Bypass -File .\setup_daily_scan.ps1 -Remove"
