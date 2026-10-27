# Dot-source from PowerShell: . .\scripts\windows\setup_env.ps1
# Process-local configuration only; does not download or install anything.
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..')).TrimEnd('\')
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME 'datasets'
$env:HF_MODULES_CACHE = Join-Path $env:HF_HOME 'modules'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
$env:CUDA_CACHE_PATH = Join-Path $projectRoot '.cache\cuda'
$env:TRITON_CACHE_DIR = Join-Path $projectRoot '.cache\triton'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $projectRoot '.cache\torchinductor'
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
foreach ($directory in @($env:HF_HUB_CACHE, $env:HF_DATASETS_CACHE, $env:HF_MODULES_CACHE, $env:TORCH_HOME, $env:PIP_CACHE_DIR, $env:UV_CACHE_DIR, $env:CUDA_CACHE_PATH, $env:TRITON_CACHE_DIR, $env:TORCHINDUCTOR_CACHE_DIR, $env:TEMP)) {
    if (-not (Test-Path -LiteralPath $directory)) {
        New-Item -ItemType Directory -Path $directory -ErrorAction Stop | Out-Null
    }
}
