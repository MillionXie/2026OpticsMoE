[CmdletBinding()]
param(
    [ValidateSet("Launch", "Foreground", "Status", "Tail", "Summarize", "Help")]
    [string]$Action = "Status",
    [string]$ServerHost = "202.120.62.181",
    [int]$Port = 24096,
    [string]$ServerUser = "guest3",
    [string]$ServerRepo = "/DATA/DATA1/guest3/2026OpticsMoE",
    [string]$PythonBin = "/home/guest3/miniconda3/envs/xml/bin/python",
    [string]$GpuList = "1,2,3,4,5",
    [string]$Seeds = "2026,2027,2028",
    [string]$AdaptationSeeds = "2026",
    [int]$PollSeconds = 20,
    [int]$MaxRetries = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

foreach ($item in @{
    GpuList = $GpuList
    Seeds = $Seeds
    AdaptationSeeds = $AdaptationSeeds
}.GetEnumerator()) {
    if ($item.Value -notmatch '^\d+(,\d+)*$') {
        throw "$($item.Key) must be a comma-separated list of non-negative integers."
    }
}
if ($PollSeconds -le 0 -or $MaxRetries -lt 0) {
    throw "PollSeconds must be positive and MaxRetries cannot be negative."
}

function ConvertTo-BashSingleQuoted([string]$Value) {
    if ($Value.Contains("'")) {
        throw "Remote path arguments cannot contain a single quote."
    }
    return "'" + $Value + "'"
}

$remoteScript = "FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/commands/p12_downstream_fa_50e.sh"
$remoteAction = $Action.ToLowerInvariant()
$repoQ = ConvertTo-BashSingleQuoted $ServerRepo
$pythonQ = ConvertTo-BashSingleQuoted $PythonBin
$gpuQ = ConvertTo-BashSingleQuoted $GpuList
$seedQ = ConvertTo-BashSingleQuoted $Seeds
$adaptationSeedQ = ConvertTo-BashSingleQuoted $AdaptationSeeds
$environment = @(
    "P12_REPO_ROOT=$repoQ",
    "P12_PYTHON_BIN=$pythonQ",
    "P12_GPU_LIST=$gpuQ",
    "P12_SEEDS=$seedQ",
    "P12_ADAPTATION_SEEDS=$adaptationSeedQ",
    "P12_POLL_SECONDS=$PollSeconds",
    "P12_MAX_RETRIES=$MaxRetries"
) -join " "
$command = "cd $repoQ && $environment bash $remoteScript $remoteAction"

Write-Host "Connecting to ${ServerUser}@${ServerHost}:${Port}; SSH may prompt for the password."
& ssh -o ServerAliveInterval=30 -p $Port "${ServerUser}@${ServerHost}" $command
if ($LASTEXITCODE -ne 0) {
    throw "Remote P12 command failed with exit code $LASTEXITCODE."
}
