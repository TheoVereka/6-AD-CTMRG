param(
    [string]$IzarHost = "chye@izar.hpc.epfl.ch",
    [string]$RemoteBundle = "~/VBCLowJ2ContinuationIzar"
)

$ErrorActionPreference = "Stop"
$files = @(
    (Join-Path $PSScriptRoot "cancel_lowJ2_D8_jobs.sh"),
    (Join-Path $PSScriptRoot "pack_completed_results.sh"),
    (Join-Path $PSScriptRoot "submit_all.sh"),
    (Join-Path $PSScriptRoot "run_one_stage.sh")
)
foreach ($file in $files) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing required file: $file"
    }
}
if ($RemoteBundle -notmatch '^[A-Za-z0-9_./~+-]+$') {
    throw "Unsafe RemoteBundle value: $RemoteBundle"
}

& scp @files "${IzarHost}:${RemoteBundle}/"
if ($LASTEXITCODE -ne 0) {
    throw "scp to Izar failed with exit code $LASTEXITCODE"
}
& ssh $IzarHost "cd ${RemoteBundle} && bash ./cancel_lowJ2_D8_jobs.sh"
if ($LASTEXITCODE -ne 0) {
    throw "Izar cancellation failed with exit code $LASTEXITCODE"
}
Write-Host "Izar low-J2 D=8 cancellation and permanent guards are installed."

