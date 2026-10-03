param(
    [string]$Remote = 'chye@izar.hpc.epfl.ch',
    [string]$RemoteBundle = '/scratch/izar/chye/connectionPlaqDimer_trees_20261003',
    [string]$DataRoot = '',
    [string]$Python = 'python',
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ($Remote -notmatch '^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$') {
    throw 'Remote must have the form user@host.'
}
if ($RemoteBundle -notmatch '^/[A-Za-z0-9_./-]+$' -or
    $RemoteBundle -match '(^|/)\.\.(/|$)') {
    throw 'RemoteBundle must be an absolute path without spaces or parent traversal.'
}

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path (Split-Path -Parent $RepoRoot) 'data'
}
$LocalBundle = Join-Path $DataRoot (Split-Path -Leaf $RemoteBundle)
$LocalResults = Join-Path $LocalBundle 'results'
$Plotter = Join-Path $PSScriptRoot 'plot_connection_energy.py'
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = $PSScriptRoot
}

foreach ($tool in @('ssh.exe', 'scp.exe', $Python)) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $tool"
    }
}
if (-not (Test-Path -LiteralPath $Plotter -PathType Leaf)) {
    throw "Missing plotter: $Plotter"
}

function Invoke-SshText {
    param([string]$Command)
    $output = @(& ssh.exe $Remote $Command)
    if ($LASTEXITCODE -ne 0) {
        throw "Izar command failed (exit $LASTEXITCODE): $Command"
    }
    return $output
}

Write-Host "Reading Izar job manifest from $RemoteBundle"
$manifestLines = @(Invoke-SshText "cat -- '$RemoteBundle/submitted_jobs.tsv'")
if ($manifestLines.Count -lt 2) {
    throw 'The remote submission manifest is empty or missing job rows.'
}
$jobs = @($manifestLines | ConvertFrom-Csv -Delimiter "`t")

# squeue contains pending, running, and completing jobs. Jobs absent from it
# have left the queue; failed jobs with a results directory are copied as well.
$activeIds = @{}
foreach ($line in @(Invoke-SshText 'squeue --noheader --user "$USER" --format "%A"')) {
    $id = $line.Trim()
    if ($id -match '^[0-9]+$') { $activeIds[$id] = $true }
}

$remoteDirs = @{}
$findCommand = "if test -d '$RemoteBundle/results'; then find '$RemoteBundle/results' -mindepth 2 -maxdepth 2 -type d -printf '%P\n'; fi"
foreach ($line in @(Invoke-SshText $findCommand)) {
    $relative = $line.Trim()
    if ($relative -match '^D[67]_(plaq|dimer)/(?:[A-I]|[O-U])$') {
        $remoteDirs[$relative] = $true
    }
}

New-Item -ItemType Directory -Path $LocalResults -Force | Out-Null
$manifestLines | Set-Content -LiteralPath (Join-Path $LocalBundle 'submitted_jobs.tsv') -Encoding UTF8

$seenIds = @{}
$statuses = @()
foreach ($job in $jobs) {
    $id = [string]$job.job_id
    $D = [string]$job.D
    $connection = [string]$job.connection
    $node = [string]$job.node
    $t = [string]$job.t
    if ($id -notmatch '^[0-9]+$' -or $seenIds.ContainsKey($id) -or
        $D -notmatch '^[67]$' -or $connection -notin @('plaq', 'dimer') -or
        $node -notmatch '^(?:[A-I]|[O-U])$' -or $t -notmatch '^[0-8]$') {
        throw "Invalid or duplicate job row: $($job | ConvertTo-Csv -NoTypeInformation | Select-Object -Last 1)"
    }
    $seenIds[$id] = $true
    $tree = "D${D}_${connection}"
    $relative = "$tree/$node"
    $state = if ($activeIds.ContainsKey($id)) {
        'active'
    } elseif ($remoteDirs.ContainsKey($relative)) {
        'ready_to_copy'
    } else {
        'ended_no_results_dir'
    }
    $statuses += [pscustomobject]@{
        job_id = $id
        D = $D
        connection = $connection
        node = $node
        t = $t
        tree = $tree
        status = $state
    }
}

$copyFailures = 0
foreach ($tree in @('D6_dimer', 'D6_plaq', 'D7_dimer', 'D7_plaq')) {
    $rows = @($statuses | Where-Object { $_.tree -eq $tree -and $_.status -eq 'ready_to_copy' })
    if ($rows.Count -eq 0) { continue }
    $destination = Join-Path $LocalResults $tree
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    $sources = @($rows | ForEach-Object { "${Remote}:${RemoteBundle}/results/$tree/$($_.node)" })
    Write-Host "Downloading $($rows.Count) finished result directories from $tree"
    & scp.exe -r @sources $destination
    if ($LASTEXITCODE -eq 0) {
        foreach ($row in $rows) { $row.status = 'downloaded' }
    } else {
        foreach ($row in $rows) { $row.status = 'copy_failed' }
        $copyFailures += $rows.Count
        Write-Warning "SCP failed for $tree (exit $LASTEXITCODE)."
    }
}

$statusPath = Join-Path $LocalBundle 'sync_status.csv'
$statuses | Export-Csv -LiteralPath $statusPath -NoTypeInformation -Encoding UTF8
Write-Host "Job download status: $statusPath"

& $Python $Plotter --data-root $LocalBundle --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Plotting failed (exit $LASTEXITCODE)."
}

$downloaded = @($statuses | Where-Object { $_.status -eq 'downloaded' }).Count
$active = @($statuses | Where-Object { $_.status -eq 'active' }).Count
$withoutDir = @($statuses | Where-Object { $_.status -eq 'ended_no_results_dir' }).Count
Write-Host "Finished result directories downloaded: $downloaded; active jobs: $active; ended without result directory: $withoutDir"
if ($copyFailures -gt 0) {
    throw "$copyFailures finished job directories could not be copied. Re-run this script to retry."
}
