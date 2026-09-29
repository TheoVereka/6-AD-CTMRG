param(
    [string]$Remote = "chye@izar.hpc.epfl.ch",
    [string]$D79RemoteDir = "~/VBCJ2SeedContinuationIzar",
    [string]$LastRemoteDir = "~/LastIzar",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$driver = Join-Path $PSScriptRoot "download_izar_d5_to_d9_and_plot.ps1"

# Compatibility wrapper with an accurate name.  The underlying driver downloads
# D=5--9 from Izar, merges the already-local Sep27/Kuma D=10,11 snapshot, and
# regenerates both fixed-D and inverse-D normal/connected NN-correlation plots.
& $driver @PSBoundParameters
