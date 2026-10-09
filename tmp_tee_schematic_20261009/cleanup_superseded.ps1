[CmdletBinding()]
param([switch]$Preview)
$ErrorActionPreference = 'Stop'
$workspace = 'D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\6-AD-CTMRG'
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$expectedTaskRoot = [IO.Path]::GetFullPath((Join-Path $workspace 'tmp_tee_schematic_20261009')).TrimEnd('\')
if (-not $taskRoot.Equals($expectedTaskRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Cleanup script is outside its exact approved workspace.'
}
if (-not (Resolve-Path -LiteralPath $workspace).ProviderPath.TrimEnd('\').Equals($workspace, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Resolved workspace does not match the approved absolute path.'
}
function Assert-NoReparseAncestors([IO.FileSystemInfo]$entry) {
    $ancestor = $entry
    while ($null -ne $ancestor) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Reparse point is not allowed: $($ancestor.FullName)"
        }
        if ($ancestor -is [IO.DirectoryInfo]) { $ancestor = $ancestor.Parent }
        else { $ancestor = $ancestor.Directory }
    }
}
function Get-ExactWorkspaceEntry([string]$relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $workspace $relative))
    if (-not $absolute.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup path escaped its approved workspace: $absolute"
    }
    if (-not (Test-Path -LiteralPath $absolute)) { return $null }
    $resolved = (Resolve-Path -LiteralPath $absolute).ProviderPath
    if (-not $resolved.Equals($absolute, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Resolved target differs from its exact approved path: $resolved"
    }
    $entry = Get-Item -LiteralPath $resolved -Force
    Assert-NoReparseAncestors $entry
    return $entry
}
function Get-TreeStats([IO.FileSystemInfo]$entry) {
    if ($null -eq $entry) { return [pscustomobject]@{ FileCount = [long]0; Bytes = [long]0 } }
    Assert-NoReparseAncestors $entry
    if (-not $entry.PSIsContainer) {
        return [pscustomobject]@{ FileCount = [long]1; Bytes = [long]$entry.Length }
    }
    $pendingDirectories = [Collections.Generic.Stack[IO.DirectoryInfo]]::new()
    $pendingDirectories.Push($entry)
    $fileCount = [long]0
    $byteCount = [long]0
    while ($pendingDirectories.Count -gt 0) {
        $directory = $pendingDirectories.Pop()
        # Inspect each child BEFORE descending; never follow any junction/link.
        foreach ($child in Get-ChildItem -LiteralPath $directory.FullName -Force) {
            if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse point inside cleanup tree: $($child.FullName)"
            }
            if ($child.PSIsContainer) { $pendingDirectories.Push($child) }
            else { $fileCount++; $byteCount += [long]$child.Length }
        }
    }
    return [pscustomobject]@{ FileCount = $fileCount; Bytes = $byteCount }
}
# User explicitly approved the WHOLE isolated archive; original source mtimes
# do not veto deletion. Only the exact targets listed below may be removed.
$approvedTargets = @(
    'tmp_tee_isolated_20261009',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\.venv_cuda',
    'tmp_tee_schematic_20261009\previous_view_v2',
    'tmp_tee_schematic_20261009\previous_view_v3',
    'tmp_tee_schematic_20261009\previous_view_v4',
    'tmp_tee_schematic_20261009\previous_view_v5',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\driver_smoke_cpu',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\driver_retry_smoke_cpu',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\driver_retry_limit_smoke_cpu',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\production_packed_smoke.json',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\production_packed_smoke.csv',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\launch_cleanup_readonly_audit.md',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\launch_validation',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\not_passed_28f694bd0c63.json',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\not_passed_28f694bd0c63.md',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\not_passed_5860f5a146be.json',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\not_passed_5860f5a146be.md',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\not_passed_c850cc14a1cc.json',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\not_passed_c850cc14a1cc.md',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\passed_a3593ff7c4a2.json',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\passed_a3593ff7c4a2.md'
)
$validationPath = Join-Path $taskRoot 'local_gpu_validation_20261009\validation_summary.json'
$validation = Get-Content -LiteralPath $validationPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($validation.gate_passed -ne $true -or $validation.full_D2_to_D6_gate_passed -ne $true) {
    throw 'Complete local D=2..6 CUDA validation must remain passed.'
}
foreach ($bondDimension in @(2, 3, 4, 5, 6)) {
    $case = @($validation.cases | Where-Object { $_.D -eq $bondDimension })
    if ($case.Count -ne 1 -or $case[0].passed -ne $true -or $case[0].status -ne 'passed') {
        throw "Local CUDA case D=$bondDimension is not recorded as passed."
    }
}
# These not_passed snapshots are only incomplete progress, not failed tests.
foreach ($relative in $approvedTargets | Where-Object { $_ -match '\\report_snapshots\\.*\.json$' }) {
    $snapshotEntry = Get-ExactWorkspaceEntry $relative
    if ($null -eq $snapshotEntry) { continue }
    $snapshot = Get-Content -LiteralPath $snapshotEntry.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($case in $snapshot.cases) {
        if ($case.status -notin @('passed', 'running', 'not_started') -or $case.error) {
            throw "Snapshot contains genuine failure evidence: $relative"
        }
    }
}
$protectedPaths = @(
    'src_code\scripts\renyi2_twoc3.py',
    'src_code\scripts\renyi2_spectral.py',
    'tmp_tee_schematic_20261009\build_schematics.py',
    'tmp_tee_schematic_20261009\depth_renderer.py',
    'tmp_tee_schematic_20261009\assemble_review.py',
    'tmp_tee_schematic_20261009\figures\four_panel_bis.png',
    'tmp_tee_schematic_20261009\figures\four_panel_bis.pdf',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\validation_summary.json',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\production_final\final_cli_validation.json',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\passed_5ecd9f5a4972.json',
    'tmp_tee_schematic_20261009\local_gpu_validation_20261009\report_snapshots\passed_5ecd9f5a4972.md'
)
$protectedHashes = @{}
foreach ($relative in $protectedPaths) {
    $entry = Get-ExactWorkspaceEntry $relative
    if ($null -eq $entry) { throw "Protected final artifact is missing: $relative" }
    $protectedHashes[$relative] = (Get-FileHash -LiteralPath $entry.FullName -Algorithm SHA256).Hash
}
# Finish path, ancestor and descendant validation for ALL targets first.
$plan = @()
foreach ($relative in $approvedTargets) {
    $entry = Get-ExactWorkspaceEntry $relative
    if ($null -eq $entry) { continue }
    $stats = Get-TreeStats $entry
    $plan += [pscustomobject]@{
        RelativePath = $relative; AbsolutePath = $entry.FullName
        IsDirectory = [bool]$entry.PSIsContainer
        FileCount = [long]$stats.FileCount; Bytes = [long]$stats.Bytes
    }
}
$totalBytes = [long](($plan | Measure-Object -Property Bytes -Sum).Sum)
$totalFiles = [long](($plan | Measure-Object -Property FileCount -Sum).Sum)
$plan | Select-Object RelativePath, FileCount, Bytes | Format-Table -AutoSize
Write-Host "Validated $($plan.Count) exact targets, $totalFiles files, $totalBytes bytes."
$logDirectory = Join-Path $taskRoot 'cleanup_logs'
if (Test-Path -LiteralPath $logDirectory) {
    Assert-NoReparseAncestors (Get-Item -LiteralPath $logDirectory -Force)
} else { New-Item -ItemType Directory -Path $logDirectory | Out-Null }
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$mode = if ($Preview) { 'preview' } else { 'executed' }
$logPath = Join-Path $logDirectory ($mode + '_' + $timestamp + '.json')
$record = [ordered]@{
    GeneratedUTC = [datetime]::UtcNow.ToString('o'); Mode = $mode; Workspace = $workspace
    Before = [ordered]@{ FileCount = $totalFiles; Bytes = $totalBytes; Targets = $plan }
    After = $null; ProtectedArtifactsUnchanged = $null; Completed = $false
}
$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $logPath -Encoding UTF8
if ($Preview) {
    $record.After = [ordered]@{ FileCount = $totalFiles; Bytes = $totalBytes; DeletedFiles = 0; DeletedBytes = 0 }
    $record.ProtectedArtifactsUnchanged = $true
    $record.Completed = $true
    $record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $logPath -Encoding UTF8
    Write-Host "Preview only; nothing deleted. Aggregate log: $logPath"
    return
}
try {
    foreach ($target in $plan) {
        $entry = Get-ExactWorkspaceEntry $target.RelativePath
        if ($null -eq $entry) { continue }
        $null = Get-TreeStats $entry
        if ($target.IsDirectory) { Remove-Item -LiteralPath $entry.FullName -Recurse -Force }
        else { Remove-Item -LiteralPath $entry.FullName -Force }
    }
    $remainingFiles = [long]0
    $remainingBytes = [long]0
    foreach ($relative in $approvedTargets) {
        $stats = Get-TreeStats (Get-ExactWorkspaceEntry $relative)
        $remainingFiles += [long]$stats.FileCount
        $remainingBytes += [long]$stats.Bytes
    }
    $record.After = [ordered]@{
        FileCount = $remainingFiles; Bytes = $remainingBytes
        DeletedFiles = $totalFiles - $remainingFiles; DeletedBytes = $totalBytes - $remainingBytes
    }
    foreach ($relative in $protectedPaths) {
        $entry = Get-ExactWorkspaceEntry $relative
        if ($null -eq $entry -or (Get-FileHash -LiteralPath $entry.FullName -Algorithm SHA256).Hash -ne $protectedHashes[$relative]) {
            throw "Protected final artifact changed unexpectedly: $relative"
        }
    }
    $record.ProtectedArtifactsUnchanged = $true
    $record.Completed = ($remainingFiles -eq 0)
    Write-Host "Removed $($record.After.DeletedFiles) obsolete files, $($record.After.DeletedBytes) bytes."
} catch {
    $record.Error = $_.Exception.Message
    throw
} finally {
    $record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $logPath -Encoding UTF8
    Write-Host "Aggregate cleanup log: $logPath"
}
