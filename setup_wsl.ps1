# Reviewable WSL core bootstrap. No distro/model/Python package download in this step.
[CmdletBinding()]
param([ValidateSet('Check', 'Core')][string]$Stage = 'Check')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
if ($projectRoot -ne 'B:\python\SlimLLM') { throw 'Run only from B:\python\SlimLLM.' }
$isAdministrator = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$diskB = [System.IO.DriveInfo]::new('B:\')
$diskC = [System.IO.DriveInfo]::new('C:\')
$state = [ordered]@{
    stage = $Stage
    administrator = $isAdministrator
    free_B_bytes = $diskB.AvailableFreeSpace
    free_C_bytes = $diskC.AvailableFreeSpace
    windows = [Environment]::OSVersion.Version.ToString()
    system_install_started = $false
}
if ($Stage -eq 'Check') { $state | ConvertTo-Json; return }
if (-not $isAdministrator) { throw 'Core requires an approved elevated PowerShell session.' }
if ($diskB.AvailableFreeSpace -lt 40GB -or $diskC.AvailableFreeSpace -lt 6GB) {
    throw 'Require B: >=40 GiB and C: >=6 GiB before system installation; do not auto-clean files.'
}
$tempRoot = Join-Path $projectRoot '.cache\tmp'
if (-not (Test-Path -LiteralPath $tempRoot)) { New-Item -ItemType Directory -Path $tempRoot | Out-Null }
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$logPath = Join-Path $tempRoot "wsl_core_$stamp.log"
$resultPath = Join-Path $projectRoot "reports\wsl_core_$stamp.json"
$wslExe = Join-Path $env:WINDIR 'System32\wsl.exe'
Start-Transcript -LiteralPath $logPath -NoClobber | Out-Null
try {
    $state['system_install_started'] = $true
    # This may enable required Windows features. It never installs a default distro.
    & $wslExe --install --no-distribution --web-download --no-launch
    $state['installer_exit_code'] = $LASTEXITCODE
    $state['reboot_executed'] = $false
    $state['note'] = 'If Windows requests a restart, stop here and let the user decide. No restart command is issued.'
    $state['log'] = $logPath
    $json = $state | ConvertTo-Json -Depth 4
    [IO.File]::WriteAllText($resultPath, $json, [Text.UTF8Encoding]::new($false))
    $json
} finally {
    Stop-Transcript | Out-Null
}
