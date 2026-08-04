param(
    [string]$EnvironmentName = "miniCamera36"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$source = Join-Path $root "ccd\lib\windows\python3.6\x64"
$runtime = Join-Path $root "dvp_runtime"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found in this PowerShell. Open Anaconda Prompt/PowerShell or initialize conda first."
}
foreach ($name in @("dvp.pyd", "DVPCamera64.dll")) {
    if (-not (Test-Path (Join-Path $source $name))) {
        throw "Missing vendor file: $(Join-Path $source $name)"
    }
}

$environmentInfo = (& conda env list --json | Out-String | ConvertFrom-Json)
$environmentPath = $environmentInfo.envs |
    Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName } |
    Select-Object -First 1

if (-not $environmentPath) {
    Write-Host "Creating isolated DVP environment $EnvironmentName (Python 3.6 x64 + NumPy)..."
    & conda create -n $EnvironmentName python=3.6 numpy=1.16.2 -y
    if ($LASTEXITCODE -ne 0) {
        throw "conda could not create $EnvironmentName"
    }
    $environmentInfo = (& conda env list --json | Out-String | ConvertFrom-Json)
    $environmentPath = $environmentInfo.envs |
        Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName } |
        Select-Object -First 1
}

$python = Join-Path $environmentPath "python.exe"
if (-not (Test-Path $python)) {
    throw "Python executable is missing after conda setup: $python"
}

New-Item -ItemType Directory -Force -Path $runtime | Out-Null
Copy-Item -Force (Join-Path $root "dvp_capture_worker.py") (Join-Path $runtime "dvp_capture_worker.py")
Copy-Item -Force (Join-Path $source "dvp.pyd") (Join-Path $runtime "dvp.pyd")
Copy-Item -Force (Join-Path $source "DVPCamera64.dll") (Join-Path $runtime "DVPCamera64.dll")

$env:DVP_PYTHON = $python
[Environment]::SetEnvironmentVariable("DVP_PYTHON", $python, "User")
$oldPath = $env:PATH
try {
    # Calling an environment's python.exe by absolute path does not perform
    # `conda activate`.  Older NumPy/MKL builds therefore need the Conda DLL
    # directories added explicitly.
    $env:PATH = "$runtime;$environmentPath;$environmentPath\Library\bin;$environmentPath\DLLs;$oldPath"
    Push-Location $runtime
    & $python -c "import struct,sys,numpy,dvp; print('python=',sys.version); print('bits=',struct.calcsize('P')*8); print('numpy=',numpy.__version__); print('dvp=',dvp.__file__)"
    if ($LASTEXITCODE -ne 0) {
        throw "DVP import verification failed"
    }
}
finally {
    Pop-Location
    $env:PATH = $oldPath
}

Write-Host ""
Write-Host "DVP runtime is ready. DVP_PYTHON=$python"
Write-Host "The user environment variable was saved permanently."
Write-Host "IMPORTANT: do NOT activate miniCamera36 to run acquire_folder.py."
Write-Host "The main acquisition program must stay on Python 3.12; only its CCD subprocess uses Python 3.6."
Write-Host "If your prompt shows (miniCamera36), run: conda deactivate"
Write-Host "If this script was dot-sourced, run acquisition in this PowerShell now."
Write-Host "Otherwise reopen PowerShell before running acquire_folder.py."
