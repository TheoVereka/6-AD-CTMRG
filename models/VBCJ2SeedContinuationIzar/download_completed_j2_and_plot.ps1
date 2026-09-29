param(
    [string]$Remote = "chye@izar.hpc.epfl.ch",
    [string]$RemoteDir = "~/VBCJ2SeedContinuationIzar",
    [string]$LocalRoot = "",
    [string]$Sep27Root = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$bundleDir = $PSScriptRoot
$repoRoot = Split-Path (Split-Path $bundleDir -Parent) -Parent
if (-not $LocalRoot) {
    $LocalRoot = [IO.Path]::GetFullPath(
        (Join-Path $repoRoot "..\data\distinVBCsJ2Continuation")
    )
}
if (-not $Sep27Root) {
    $Sep27Root = [IO.Path]::GetFullPath(
        (Join-Path $repoRoot "..\data\external\Working_AD_Honeycomb_Sep27")
    )
}
$archiveName = "Izar_completed_J2_continuation.tar.gz"
$archive = Join-Path $LocalRoot $archiveName
$packer = Join-Path $bundleDir "pack_completed_j2_stages.sh"
$plotter = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\plot_j2_seed_continuations.py"
)
$manifest = Join-Path $bundleDir "selected_seed_manifest.csv"
$inputRoot = Join-Path $LocalRoot "Results_Izar_J2_sequences"
$plotRoot = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\j2_seed_continuations"
)
$sep27PlotRoot = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\sep27_j2_seed_continuations"
)

if (-not (Test-Path -LiteralPath $Sep27Root -PathType Container)) {
    throw "Sep27 result root is missing: $Sep27Root"
}
if (-not (Test-Path -LiteralPath (Join-Path $Sep27Root "private_manifest.tsv") -PathType Leaf)) {
    throw "Sep27 private_manifest.tsv is missing under: $Sep27Root"
}

New-Item -ItemType Directory -Path $LocalRoot -Force | Out-Null

Write-Host "Uploading the completed-stage packer..."
& scp $packer "${Remote}:${RemoteDir}/pack_completed_j2_stages.sh"
if ($LASTEXITCODE -ne 0) { throw "scp upload failed ($LASTEXITCODE)" }

Write-Host "Creating an Izar snapshot of individually completed J2 stages..."
& ssh $Remote "cd ${RemoteDir} && bash pack_completed_j2_stages.sh"
if ($LASTEXITCODE -ne 0) { throw "remote pack failed ($LASTEXITCODE)" }

Write-Host "Downloading $archiveName..."
& scp "${Remote}:${RemoteDir}/${archiveName}" $archive
if ($LASTEXITCODE -ne 0) { throw "scp download failed ($LASTEXITCODE)" }

# Extraction is additive: a later snapshot updates/adds completed stages and
# does not erase results downloaded from an earlier snapshot.
Write-Host "Merging the snapshot into $LocalRoot..."
& tar -xzf $archive -C $LocalRoot
if ($LASTEXITCODE -ne 0) { throw "archive extraction failed ($LASTEXITCODE)" }

Write-Host "Plotting original D=5,6 plus accumulated Izar D=7,8,9 stages..."
& $Python $plotter --input $inputRoot --manifest $manifest --output-dir $plotRoot
if ($LASTEXITCODE -ne 0) { throw "plotting failed ($LASTEXITCODE)" }

Write-Host "Plotting current Sep27 D=10,11 stages and updating combined D=5--11 fits..."
& $Python $plotter --input $Sep27Root --output-dir $sep27PlotRoot
if ($LASTEXITCODE -ne 0) { throw "Sep27/combined plotting failed ($LASTEXITCODE)" }

Write-Host "Fixed-D D=5--9 plots: $plotRoot"
Write-Host "Fixed-D D=10--11 plots: $sep27PlotRoot"
Write-Host "Combined inverse-D D=5--11 fits: $plotRoot"
