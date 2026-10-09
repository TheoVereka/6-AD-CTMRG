param(
    [string]$IzarHost = "chye@izar.hpc.epfl.ch",
    [string]$RemoteBundle = "~/VBCLowJ2ContinuationIzar",
    [string]$ExternalRoot = "",
    [string]$KumaResults = "",
    [string]$KumaExtendedResults = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Native command failed ($LASTEXITCODE): $Program $($Arguments -join ' ')"
    }
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..")).Path
if ([string]::IsNullOrWhiteSpace($ExternalRoot)) {
    $ExternalRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "..\data\external"))
}
if ([string]::IsNullOrWhiteSpace($KumaResults)) {
    $KumaResults = Join-Path $ExternalRoot "VBCLowJ2ContinuationKuma\Results_Kuma_lowJ2"
}
if ([string]::IsNullOrWhiteSpace($KumaExtendedResults)) {
    $KumaExtendedResults = Join-Path $ExternalRoot "VBCLowJ2ContinuationKumaExtended\Results_Kuma_lowJ2_extended"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $PSScriptRoot "plots"
}

$packer = Join-Path $repoRoot "models\VBCLowJ2ContinuationIzar\pack_completed_results.sh"
$plotter = Join-Path $PSScriptRoot "plot_low_j2_pinning_continuation.py"
$izarLocal = Join-Path $ExternalRoot "VBCLowJ2ContinuationIzar"
$izarResults = Join-Path $izarLocal "Results_Izar_lowJ2"
$archive = Join-Path $izarLocal "Izar_lowJ2_completed.tar.gz"
$archivePart = "$archive.part"

if (-not (Test-Path -LiteralPath $packer -PathType Leaf)) {
    throw "Missing local packer: $packer"
}
if (-not (Test-Path -LiteralPath $plotter -PathType Leaf)) {
    throw "Missing plotter: $plotter"
}
if (-not (Test-Path -LiteralPath $KumaResults -PathType Container)) {
    throw "Kuma result root not found: $KumaResults"
}
if ($RemoteBundle -notmatch '^[A-Za-z0-9_./~+-]+$') {
    throw "Unsafe RemoteBundle value: $RemoteBundle"
}

New-Item -ItemType Directory -Force -Path $izarLocal | Out-Null
New-Item -ItemType Directory -Force -Path $Output | Out-Null

Write-Host "[1/4] Uploading the completed-stage packer to Izar..."
Invoke-NativeChecked scp $packer "${IzarHost}:${RemoteBundle}/pack_completed_results.sh"

Write-Host "[2/4] Creating an atomic snapshot of completed Izar stages..."
Invoke-NativeChecked ssh $IzarHost "cd ${RemoteBundle} && bash ./pack_completed_results.sh"

Write-Host "[3/4] Downloading and expanding the completed-stage snapshot..."
Invoke-NativeChecked scp "${IzarHost}:${RemoteBundle}/Izar_lowJ2_completed.tar.gz" $archivePart
Move-Item -LiteralPath $archivePart -Destination $archive -Force
Invoke-NativeChecked tar -xzf $archive -C $izarLocal

Write-Host "[4/4] Combining Izar + Kuma and plotting every requested signed h..."
$plotArguments = @($plotter, "--kuma", $KumaResults)
if (Test-Path -LiteralPath $KumaExtendedResults -PathType Container) {
    $plotArguments += @("--kuma", $KumaExtendedResults)
    Write-Host "Including extended Kuma results: $KumaExtendedResults"
}
$plotArguments += @("--izar", $izarResults, "--output", $Output)
Invoke-NativeChecked python @plotArguments

Write-Host "Done. Plots and audit CSV files are in: $Output"

