param(
    [string]$Remote = "chye@izar.hpc.epfl.ch",
    [string]$D79RemoteDir = "~/VBCJ2SeedContinuationIzar",
    [string]$LastRemoteDir = "~/LastIzar",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$lastBundle = $PSScriptRoot
$repoRoot = Split-Path (Split-Path $lastBundle -Parent) -Parent
$d79Bundle = Join-Path $repoRoot "models\VBCJ2SeedContinuationIzar"
$dataRoot = [IO.Path]::GetFullPath(
    (Join-Path $repoRoot "..\data\distinVBCsJ2Continuation")
)
$d79Input = Join-Path $dataRoot "Results_Izar_J2_sequences"
$lastLocal = Join-Path $dataRoot "LastIzar"
$plotRoot = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\j2_seed_continuations"
)
$sep27Root = [IO.Path]::GetFullPath(
    (Join-Path $repoRoot "..\data\external\Working_AD_Honeycomb_Sep27")
)
$sep27PlotRoot = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\sep27_j2_seed_continuations"
)
$plotter = Join-Path $repoRoot (
    "visual_elements\figs\VBCDiscriminator\plot_j2_seed_continuations.py"
)
$diagnoser = Join-Path $lastBundle "diagnose_downloaded_results.py"
$manifest = Join-Path $d79Bundle "selected_seed_manifest.csv"

New-Item -ItemType Directory -Force -Path $dataRoot,$lastLocal,$plotRoot | Out-Null

# D=7,8,9: package only individually completed J2 stages.
$d79Packer = Join-Path $d79Bundle "pack_completed_j2_stages.sh"
Write-Host "[1/7] Updating completed Izar D=7,8,9 stages..."
& scp $d79Packer "${Remote}:${D79RemoteDir}/pack_completed_j2_stages.sh"
if ($LASTEXITCODE -ne 0) { throw "D7--9 packer upload failed ($LASTEXITCODE)" }
& ssh $Remote "cd ${D79RemoteDir} && bash pack_completed_j2_stages.sh"
if ($LASTEXITCODE -ne 0) { throw "D7--9 remote packing failed ($LASTEXITCODE)" }
$d79Archive = Join-Path $dataRoot "Izar_completed_J2_continuation.tar.gz"
& scp "${Remote}:${D79RemoteDir}/Izar_completed_J2_continuation.tar.gz" $d79Archive
if ($LASTEXITCODE -ne 0) { throw "D7--9 download failed ($LASTEXITCODE)" }
& tar -xzf $d79Archive -C $dataRoot
if ($LASTEXITCODE -ne 0) { throw "D7--9 extraction failed ($LASTEXITCODE)" }

# D=5,6: snapshot results, logs, squeue, sacct and dependency records.
$lastPacker = Join-Path $lastBundle "make_download_snapshot.sh"
Write-Host "[2/7] Snapshotting partial/completed LastIzar D=5,6 data and logs..."
& scp $lastPacker "${Remote}:${LastRemoteDir}/make_download_snapshot.sh"
if ($LASTEXITCODE -ne 0) { throw "LastIzar packer upload failed ($LASTEXITCODE)" }
& ssh $Remote "cd ${LastRemoteDir} && bash make_download_snapshot.sh"
if ($LASTEXITCODE -ne 0) { throw "LastIzar remote snapshot failed ($LASTEXITCODE)" }
$lastArchive = Join-Path $lastLocal "LastIzar_results_snapshot.tar.gz"
& scp "${Remote}:${LastRemoteDir}/LastIzar_results_snapshot.tar.gz" $lastArchive
if ($LASTEXITCODE -ne 0) { throw "LastIzar download failed ($LASTEXITCODE)" }

# h=.005 task2--task5 were explicitly retired.  Tar extraction does not
# remove directories that are absent from a newer archive, so clear only
# these exact obsolete local targets before unpacking the new snapshot.
$lastResultRoot = [IO.Path]::GetFullPath(
    (Join-Path $lastLocal "Results_LastIzar")
)
foreach ($obsolete in @(
    "task2_D6_adam_pin", "task3_D6_lbfgs_pin",
    "task4_D5_adam_pin", "task5_D5_lbfgs_pin"
)) {
    $target = [IO.Path]::GetFullPath((Join-Path $lastResultRoot $obsolete))
    if (-not $target.StartsWith(
        $lastResultRoot + [IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Unsafe obsolete-result target: $target"
    }
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}
$lastLogRoot = Join-Path $lastLocal "slurm_logs"
if (Test-Path -LiteralPath $lastLogRoot) {
    Get-ChildItem -LiteralPath $lastLogRoot -File |
        Where-Object { $_.Name -match '^L[2345]' } |
        Remove-Item -Force
}
& tar -xzf $lastArchive -C $lastLocal
if ($LASTEXITCODE -ne 0) { throw "LastIzar extraction failed ($LASTEXITCODE)" }

Write-Host "[3/7] Diagnosing D=5,6 jobs and afterok failures..."
& $Python $diagnoser --snapshot $lastLocal --output-dir $plotRoot
if ($LASTEXITCODE -ne 0) { throw "LastIzar diagnosis failed ($LASTEXITCODE)" }

Write-Host "[4/7] Plotting fixed-D normal and connected NN correlations (D=5--9)..."
$fixedDArgs = @(
    $plotter, "--input", $d79Input, "--manifest", $manifest,
    "--last-izar-input", (Join-Path $lastLocal "Results_LastIzar"),
    "--output-dir", $plotRoot
)
& $Python @fixedDArgs
if ($LASTEXITCODE -ne 0) { throw "D5--9 plotting failed ($LASTEXITCODE)" }

Write-Host "[5/7] Plotting Sep27 D=10,11 and refreshing all inverse-D figures..."
if (Test-Path -LiteralPath (Join-Path $sep27Root "private_manifest.tsv")) {
    $sep27Args = @(
        $plotter, "--input", $sep27Root,
        "--last-izar-input", (Join-Path $lastLocal "Results_LastIzar"),
        "--output-dir", $sep27PlotRoot
    )
    & $Python @sep27Args
    if ($LASTEXITCODE -ne 0) { throw "D10--11/inverse-D plotting failed ($LASTEXITCODE)" }
} else {
    Write-Warning "Sep27 data absent; skipping D=10,11 and combined inverse-D refresh"
}

Write-Host "[6/7] Plot inventory:"
$normal = @(Get-ChildItem -LiteralPath $plotRoot -Filter '2C3_NN_ranks_vs_J2_D*.pdf')
$connected = @(Get-ChildItem -LiteralPath $plotRoot -Filter '2C3_connected_NN_ranks_vs_J2_D*.pdf')
$inverse = @(Get-ChildItem -LiteralPath $plotRoot -Filter '2C3_VBC_NN_ranks_vs_inverse_D_J2_*.pdf')
$inverseConnected = @(Get-ChildItem -LiteralPath $plotRoot -Filter '2C3_VBC_connected_NN_ranks_vs_inverse_D_J2_*.pdf')
Write-Host "  fixed-D normal: $($normal.Count)"
Write-Host "  fixed-D connected: $($connected.Count)"
Write-Host "  inverse-D normal: $($inverse.Count)"
Write-Host "  inverse-D connected: $($inverseConnected.Count)"

Write-Host "[7/7] Complete."
Write-Host "Plots/diagnostics: $plotRoot"
Write-Host "D=7--9 raw snapshot: $dataRoot"
Write-Host "D=5--6 raw snapshot/logs: $lastLocal"
