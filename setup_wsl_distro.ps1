# Separate approval is required before -Stage Install. Check is read-only.
[CmdletBinding()]
param([ValidateSet('Check', 'Install')][string]$Stage = 'Check')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
if ($projectRoot -ne 'B:\python\SlimLLM') { throw 'Run only from B:\python\SlimLLM.' }
$distroName = 'SlimLLM-Ubuntu'
$distroRoot = Join-Path $projectRoot '.wsl\Ubuntu'
$tempRoot = Join-Path $projectRoot '.cache\tmp'
$configPath = Join-Path $env:USERPROFILE '.wslconfig'
$configText = @'
[wsl2]
memory=8GB
processors=4
swap=2GB
swapfile=B:\\python\\SlimLLM\\.cache\\wsl-swap.vhdx
localhostForwarding=true
'@
$configText = $configText.Replace("`r`n", "`n").Trim() + "`n"
$diskB = [IO.DriveInfo]::new('B:\')
$diskC = [IO.DriveInfo]::new('C:\')
$registryRoot = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss'
$distributions = @()
if (Test-Path -LiteralPath $registryRoot) {
    $distributions = @(Get-ChildItem -LiteralPath $registryRoot | ForEach-Object {
        (Get-ItemProperty -LiteralPath $_.PSPath).DistributionName
    })
}
$installArgs = @('--install', 'Ubuntu-24.04', '--name', $distroName,
    '--location', $distroRoot, '--version', '2', '--web-download', '--no-launch')
$state = [ordered]@{
    stage = $Stage
    distribution = $distroName
    distribution_location = $distroRoot
    registered_distributions = $distributions
    global_config_path = $configPath
    global_config_exists = (Test-Path -LiteralPath $configPath)
    global_config_proposed = $configText
    free_B_bytes = $diskB.AvailableFreeSpace
    free_C_bytes = $diskC.AvailableFreeSpace
    install_arguments = $installArgs
    config_created = $false
    installation_started = $false
    distro_launched_by_script = $false
    linux_dependencies_installed_by_script = $false
    reboot_executed = $false
}
if ($Stage -eq 'Check') { $state | ConvertTo-Json -Depth 5; return }
if ($diskB.AvailableFreeSpace -lt 40GB -or $diskC.AvailableFreeSpace -lt 6GB) {
    throw 'Require B: >=40 GiB and C: >=6 GiB. Do not auto-clean files.'
}
if ($distributions.Count -ne 0) {
    throw 'A distribution now exists. Re-review global settings and registration; do not replace or shut it down.'
}
if (Test-Path -LiteralPath $distroRoot) {
    throw 'The distribution target already exists. Inspect a previous attempt; never overwrite or delete it automatically.'
}
if (Test-Path -LiteralPath $configPath) {
    $existing = [IO.File]::ReadAllText($configPath).Replace("`r`n", "`n").Trim() + "`n"
    if ($existing -cne $configText) { throw 'A different global .wslconfig exists; do not overwrite it.' }
}
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $distroRoot) -Force | Out-Null
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$logPath = Join-Path $tempRoot "wsl_distro_$stamp.log"
$resultPath = Join-Path $projectRoot "reports\wsl_distro_$stamp.json"
$wslExe = Join-Path $env:WINDIR 'System32\wsl.exe'
Start-Transcript -LiteralPath $logPath -NoClobber | Out-Null
try {
    if (-not (Test-Path -LiteralPath $configPath)) {
        # CreateNew prevents accidental overwrite if another process creates it.
        $stream = [IO.File]::Open($configPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        try {
            $bytes = [Text.UTF8Encoding]::new($false).GetBytes($configText)
            $stream.Write($bytes, 0, $bytes.Length)
        } finally { $stream.Dispose() }
        $state['config_created'] = $true
    }
    $state['installation_started'] = $true
    & $wslExe @installArgs
    $state['installer_exit_code'] = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw "WSL distribution installer returned $LASTEXITCODE. Inspect log before retrying." }
} catch {
    $state['error'] = $_.Exception.Message
    throw
} finally {
    $state['log'] = $logPath
    $state['free_B_bytes_after'] = ([IO.DriveInfo]::new('B:\')).AvailableFreeSpace
    $state['free_C_bytes_after'] = ([IO.DriveInfo]::new('C:\')).AvailableFreeSpace
    $json = $state | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText($resultPath, $json, [Text.UTF8Encoding]::new($false))
    $json
    Stop-Transcript | Out-Null
}
