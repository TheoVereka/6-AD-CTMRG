param(
    [string]$Remote = "chye@izar.hpc.epfl.ch",
    [string]$RemoteDir = "~/LastIzarD7Dimer026",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$bundle = $PSScriptRoot
$repo = Split-Path (Split-Path $bundle -Parent) -Parent
$target = [IO.Path]::GetFullPath(
    (Join-Path $repo "..\data\distinVBCsJ2Continuation\D7DimerJ2_0p26")
)
New-Item -ItemType Directory -Force -Path $target | Out-Null
& scp (Join-Path $bundle "make_download_snapshot.sh") "${Remote}:${RemoteDir}/make_download_snapshot.sh"
if ($LASTEXITCODE -ne 0) { throw "snapshot script upload failed" }
& ssh $Remote "cd ${RemoteDir} && bash make_download_snapshot.sh"
if ($LASTEXITCODE -ne 0) { throw "remote snapshot failed" }
$archive = Join-Path $target "D7DimerJ2_0p26_snapshot.tar.gz"
& scp "${Remote}:${RemoteDir}/D7DimerJ2_0p26_snapshot.tar.gz" $archive
if ($LASTEXITCODE -ne 0) { throw "snapshot download failed" }
& tar -xzf $archive -C $target
if ($LASTEXITCODE -ne 0) { throw "snapshot extraction failed" }
Write-Host "D7/J2=.26 snapshot updated: $target"
$selector = Join-Path $repo (
    "visual_elements\figs\VBCDiscriminator\selected_nn_story\select_and_plot.py"
)
& $Python $selector
if ($LASTEXITCODE -ne 0) { throw "selected NN plotting failed" }
