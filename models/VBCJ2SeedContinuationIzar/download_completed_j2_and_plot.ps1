param(
    [string]$Remote = "chye@izar.hpc.epfl.ch",
    [string]$RemoteDir = "~/VBCJ2SeedContinuationIzar",
    [string]$LocalRoot = "",
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
$archiveName = "Izar_completed_J2_continuation.tar.gz"
$archive = Join-Path $LocalRoot $archiveName
$packer = Join-Path $bundleDir "pack_completed_j2_stages.sh"
$plotter = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\plot_j2_seed_continuations.py"
)
$extrapolator = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\plot_vbc_seed_inverse_D_extrapolations.py"
)
$manifest = Join-Path $bundleDir "selected_seed_manifest.csv"
$inputRoot = Join-Path $LocalRoot "Results_Izar_J2_sequences"
$plotRoot = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\j2_seed_continuations"
)

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

Write-Host "Plotting all completed stages accumulated locally..."
& $Python $plotter --input $inputRoot --manifest $manifest --output-dir $plotRoot
if ($LASTEXITCODE -ne 0) { throw "plotting failed ($LASTEXITCODE)" }

Write-Host "Updating two-panel VBC-seed D=7,8,9 extrapolations..."
& $Python $extrapolator --input $inputRoot --manifest $manifest --output-dir $plotRoot
if ($LASTEXITCODE -ne 0) { throw "extrapolation plotting failed ($LASTEXITCODE)" }

Write-Host "Plots: $plotRoot"
