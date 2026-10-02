param(
    [string]$InputRoot = "D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\distinVBCsKumaTargetedRepairs\Results_Kuma_TargetedRepairs"
)

$ErrorActionPreference = "Stop"
$BundleDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $BundleDir "..\..")).Path
$ExpectedRoot = (Join-Path (Split-Path -Parent $RepoRoot) "data\distinVBCsKumaTargetedRepairs\Results_Kuma_TargetedRepairs")

if (-not (Test-Path -LiteralPath $InputRoot -PathType Container)) {
    throw "Downloaded Kuma result folder is missing: $InputRoot"
}
if ([IO.Path]::GetFullPath($InputRoot).TrimEnd('\') -ne [IO.Path]::GetFullPath($ExpectedRoot).TrimEnd('\')) {
    throw "Place the download at the fixed analysis path: $ExpectedRoot"
}

$Targets = @(
    @{ Alias="r01"; D=10; Chi=120 },
    @{ Alias="r02"; D=10; Chi=120 },
    @{ Alias="r03"; D=11; Chi=140 }
)
foreach ($Target in $Targets) {
    foreach ($Stage in @("h_0p02", "h_0")) {
        $StageDir = Join-Path $InputRoot (Join-Path $Target.Alias $Stage)
        $Required = @(
            (Join-Path $StageDir "COMPLETED.stage"),
            (Join-Path $StageDir ("D_{0}_chi_{1}_energy_magnetization_correlation.txt" -f $Target.D,$Target.Chi)),
            (Join-Path $StageDir ("sweep_D{0}_chi{1}_best.pt" -f $Target.D,$Target.Chi)),
            (Join-Path $StageDir "hyperparams.yaml")
        )
        foreach ($Path in $Required) {
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                throw "Incomplete repair stage; missing $Path"
            }
        }
    }
}

Push-Location $RepoRoot
try {
    python .\visual_elements\figs\VBCDiscriminator\selected_nn_story\select_and_plot.py
    if ($LASTEXITCODE -ne 0) { throw "select_and_plot.py failed with exit code $LASTEXITCODE" }

    python .\visual_elements\figs\VBCDiscriminator\selected_nn_story\pinning_energy_phase_boundary.py
    if ($LASTEXITCODE -ne 0) { throw "pinning_energy_phase_boundary.py failed with exit code $LASTEXITCODE" }

    python .\visual_elements\figs\VBCDiscriminator\selected_nn_story\plot_crossing_energy_comparison.py
    if ($LASTEXITCODE -ne 0) { throw "plot_crossing_energy_comparison.py failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "Refreshed selected NN story. E_crossing,D received its error-weighted gapped extrapolation; h_c,D was statistically combined by uncertainty and energy proximity; all dependent plots were regenerated."
