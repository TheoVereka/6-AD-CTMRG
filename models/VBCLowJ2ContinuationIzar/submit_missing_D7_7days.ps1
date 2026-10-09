param(
    [string]$IzarHost = "chye@izar.hpc.epfl.ch",
    [string]$RemoteBundle = "~/VBCLowJ2ContinuationIzar"
)

$ErrorActionPreference = "Stop"
$files = @(
    (Join-Path $PSScriptRoot "submit_missing_D7_7days.sh"),
    (Join-Path $PSScriptRoot "run_one_stage.sh")
)
foreach ($file in $files) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing recovery file: $file"
    }
}
if ($RemoteBundle -notmatch '^[A-Za-z0-9_./~+-]+$') {
    throw "Unsafe RemoteBundle value: $RemoteBundle"
}

& scp @files "${IzarHost}:${RemoteBundle}/"
if ($LASTEXITCODE -ne 0) {
    throw "scp to Izar failed with exit code $LASTEXITCODE"
}

& ssh $IzarHost "cd ${RemoteBundle} && bash ./submit_missing_D7_7days.sh --dry-run && bash ./submit_missing_D7_7days.sh"
if ($LASTEXITCODE -ne 0) {
    throw "Izar recovery submission failed with exit code $LASTEXITCODE"
}
Write-Host "Submitted the nine missing D=7 stages on seven-day QOS."

