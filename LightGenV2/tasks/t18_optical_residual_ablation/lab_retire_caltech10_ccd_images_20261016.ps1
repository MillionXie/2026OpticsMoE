param([ValidateSet('SelfTest','Inspect','Prepare','Delete','Verify')][string]$Mode='Inspect',[string]$ManifestSHA='')
$ErrorActionPreference='Stop'
function IsCCDName([string]$Name) { return $Name -cmatch '^\d{6}(_raw)?\.png$' }
if ($Mode -eq 'SelfTest') {
    foreach ($name in @('000000.png','001832_raw.png')) { if (-not (IsCCDName $name)) { throw 'Positive filename test failed' } }
    foreach ($name in @('000000_input.png','phase.bmp','ccd.npy','response.png','../000000.png','000000_raw.jpg')) { if (IsCCDName $name) { throw 'Protected filename test failed' } }
    Write-Output 'CCD leaf-name allowlist tests passed; no files touched'; exit 0
}
$Repo='E:\code\guest\2026OpticsMoE'
$Task=Join-Path $Repo 'LightGenV2\tasks\t18_optical_residual_ablation'
$ExpectedRoot=Join-Path $Task 'runs\hardware\caltech10_L3_20261015'
$Root=(Resolve-Path -LiteralPath $ExpectedRoot).ProviderPath.TrimEnd('\')
if (-not $Root.Equals($ExpectedRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected resolved session root' }
$Archive=Join-Path $Root 'ccd_image_retirement_20261016'
$Manifest=Join-Path $Archive 'original_images.psv'
$Groups=@(
    @{rel='full_train_20261016';depth=3;n=761},@{rel='full_test_20261016';depth=3;n=1833},
    @{rel='depth_L2L4_20261016\L2\train';depth=2;n=761},@{rel='depth_L2L4_20261016\L2\test';depth=2;n=1833},
    @{rel='depth_L2L4_20261016\L4\train';depth=4;n=761},@{rel='depth_L2L4_20261016\L4\test';depth=4;n=1833}
)
$AllowedParents=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($group in $Groups) {
    $path=(Resolve-Path -LiteralPath (Join-Path $Root $group.rel)).ProviderPath
    if (-not $path.StartsWith($Root+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Capture folder escaped explicit session root' }
    $report=Get-Content -LiteralPath (Join-Path $path 'report.json') -Raw | ConvertFrom-Json
    if ($report.state -ne 'complete' -or -not $report.sdk_released -or $report.physical_ccd -ne 2*$group.n*($group.depth+1)) { throw 'Incomplete capture report' }
    foreach ($arm in @('residual','control')) { foreach ($stage in (@('router')+@(0..($group.depth-1) | ForEach-Object { [string]$_ }))) {
        $dir=(Resolve-Path -LiteralPath (Join-Path (Join-Path $path $arm) $stage)).ProviderPath
        $ancestor=$dir
        while ($ancestor.Length -ge $Root.Length) {
            if ((Get-Item -LiteralPath $ancestor).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point capture ancestor prohibited' }
            $ancestor=[IO.Path]::GetDirectoryName($ancestor)
        }
        [void]$AllowedParents.Add($dir)
    } }
}
function AssertTarget([string]$Path) {
    $full=[IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($Root+'\',[StringComparison]::OrdinalIgnoreCase) -or -not $AllowedParents.Contains([IO.Path]::GetDirectoryName($full)) -or -not (IsCCDName ([IO.Path]::GetFileName($full)))) { throw "Target outside exact image allowlist: $Path" }
    if ((Get-Item -LiteralPath $full).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Image reparse point prohibited' }
    return $full
}
function Guards {
    if (Test-Path -LiteralPath (Join-Path $Task 'runs\hardware\.t18_sdk_owner.lock')) { throw 'SDK owner still present' }
    $pattern='lab_(caltech10_20261015|depth_capture_20261016|chain_20261016|readout_20261016|wait_readout_20261016|saved_train_audit_20261016)\.py'
    $active=@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match $pattern -or $_.CommandLine -like '*caltech10_L3_20261015*') })
    if ($active.Count) { throw 'A process still references the old capture/session' }
    $service=New-Object -ComObject Schedule.Service; $service.Connect(); $states=@()
    # These on-demand experiment jobs are hidden: TASK_ENUM_HIDDEN=1 is required.
    foreach ($job in $service.GetFolder('\').GetTasks(1)) {
        $matches=$false
        foreach ($action in $job.Definition.Actions) { if ($action.Arguments -match $pattern -or $action.Arguments -like '*caltech10_L3_20261015*') { $matches=$true } }
        if ($matches) {
            $states+=@{name=$job.Name;state=$job.State;triggers=$job.Definition.Triggers.Count;last_result=$job.LastTaskResult}
            if ($job.State -eq 4 -or $job.Definition.Triggers.Count -gt 0) { throw 'Running or future-triggered old task dependency' }
        }
    }
    return $states
}
function Candidates {
    foreach ($parent in $AllowedParents) { Get-ChildItem -LiteralPath $parent -File | Where-Object { IsCCDName $_.Name } }
}
$states=@(Guards)
$git=Join-Path $Repo 'OpenMoji_Robust_Rank64_SHS_20261002\.git-tools\MinGit_2_56_0\cmd\git.exe'
$refs=@(& $git -C $Repo grep -l -F 'caltech10_L3_20261015' -- 'LightGenV2' ':(exclude)**/runs/**')
if ($LASTEXITCODE -gt 1) { throw 'Cross-reference inventory failed' }
$stats=@{files=0L;bytes=0L}; Candidates | ForEach-Object { if ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Image reparse point prohibited' };$stats.files++;$stats.bytes+=$_.Length }
$npy=@{files=0L;bytes=0L};foreach ($parent in $AllowedParents) { $p=Join-Path $parent 'ccd.npy';if (Test-Path -LiteralPath $p) { $npy.files++;$npy.bytes+=(Get-Item -LiteralPath $p).Length } }
$base=@{root=$Root;mode=$Mode;candidate_images=$stats;pixel_memmaps_preserved=$npy;scheduled_dependencies=$states;source_references=$refs;authorization_thread='01a1201b-0c04-7741-a18a-fb708323c960';authorization='Human2026-10-16 explicitly permits deleting old Caltech10 CCD images; no memmap deletion';calibration_pilots_inputs_weights_receipts_features_predictions_preserved=$true}
if ($Mode -eq 'Inspect') { $base | ConvertTo-Json -Depth 8 -Compress; exit 0 }
if ($Mode -eq 'Prepare') {
    if (Test-Path -LiteralPath $Archive) { throw 'Existing retirement inventory; inspect rather than overwrite' }
    [void](New-Item -ItemType Directory -Path $Archive)
    $writer=[IO.StreamWriter]::new($Manifest,$false,[Text.UTF8Encoding]::new($false));$counter=0L
    try {
        $writer.WriteLine('absolute_path|bytes|mtime_ticks|sha256')
        Candidates | ForEach-Object {
            $full=AssertTarget $_.FullName;$hash=(Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLowerInvariant()
            $writer.WriteLine($full+'|'+$_.Length+'|'+$_.LastWriteTimeUtc.Ticks+'|'+$hash);$counter++
            if ($counter%1000 -eq 0) { $writer.Flush(); $progress=@{hashed_images=$counter;target=$stats.files;time_utc=[DateTime]::UtcNow.ToString('o')};$progress | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $Archive ('prepare_progress_'+$counter.ToString('D6')+'.json')) -Encoding UTF8 }
        }
    } catch { @{state='preparation_failed_no_deletion';error=$_.Exception.Message;hashed=$counter} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Archive 'prepare_failure.json') -Encoding UTF8;throw } finally { $writer.Dispose() }
    if ($counter -ne $stats.files) { throw 'Image count changed while preparing' }
    $base.manifest_sha256=(Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    $base.prepared_utc=[DateTime]::UtcNow.ToString('o');$base.source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $base | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Archive 'prepared.json') -Encoding UTF8
    $base | ConvertTo-Json -Depth 8 -Compress; exit 0
}
if ($Mode -eq 'Delete') {
    $prepared=Get-Content -LiteralPath (Join-Path $Archive 'prepared.json') -Raw | ConvertFrom-Json
    $actual=(Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ManifestSHA -notmatch '^[a-f0-9]{64}$' -or $actual -ne $ManifestSHA -or $actual -ne $prepared.manifest_sha256) { throw 'Explicit SHA-bound manifest required' }
    if ($stats.files -ne $prepared.candidate_images.files -or $stats.bytes -ne $prepared.candidate_images.bytes) { throw 'Inventory changed; do not delete' }
    $backup=Get-Content -LiteralPath (Join-Path $Archive 'inventory_local_backup_verified.json') -Raw | ConvertFrom-Json
    if ($backup.manifest_sha256 -ne $actual -or -not $backup.local_verified) { throw 'Independent local metadata backup not verified' }
    $log=Join-Path $Archive 'deleted_images.psv';if (Test-Path -LiteralPath $log) { throw 'Deletion already attempted; inspect partial evidence, no blind retry' }
    $freeBefore=[IO.DriveInfo]::new('E:\').AvailableFreeSpace;$tally=@{files=0L;bytes=0L};$writer=[IO.StreamWriter]::new($log,$false,[Text.UTF8Encoding]::new($false))
    try {
        $writer.WriteLine('absolute_path|bytes|sha256|deleted_utc')
        Import-Csv -LiteralPath $Manifest -Delimiter '|' | ForEach-Object {
            $full=AssertTarget $_.absolute_path;$item=Get-Item -LiteralPath $full
            if ($item.Length -ne [long]$_.bytes -or $item.LastWriteTimeUtc.Ticks -ne [long]$_.mtime_ticks) { throw 'Image changed after immutable inventory' }
            # Each exact validated leaf only: no recursive or wildcard deletion.
            Remove-Item -LiteralPath $full -ErrorAction Stop
            if (Test-Path -LiteralPath $full) { throw 'Image remained after deletion' }
            $tally.files++;$tally.bytes+=[long]$_.bytes
            $writer.WriteLine($full+'|'+$_.bytes+'|'+$_.sha256+'|'+[DateTime]::UtcNow.ToString('o'))
            if ($tally.files%1000 -eq 0) { $writer.Flush();@{deleted_images=$tally.files;target=$prepared.candidate_images.files;deleted_bytes=$tally.bytes;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $Archive ('delete_progress_'+$tally.files.ToString('D6')+'.json')) -Encoding UTF8 }
        }
    } catch {
        @{state='partial_failure_no_automatic_retry';error=$_.Exception.Message;deleted=$tally;time_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Archive 'failure.json') -Encoding UTF8
        throw
    } finally { $writer.Dispose() }
    $freeAfter=[IO.DriveInfo]::new('E:\').AvailableFreeSpace
    $result=@{state='complete';deleted=$tally;disk_free_before=$freeBefore;disk_free_after=$freeAfter;observed_disk_free_change=$freeAfter-$freeBefore;manifest_sha256=$actual;deletion_log_sha256=(Get-FileHash -LiteralPath $log -Algorithm SHA256).Hash.ToLowerInvariant();completed_utc=[DateTime]::UtcNow.ToString('o');recoverable_image_backup_created=$false;original_images_still_readable=$false;pixel_memmaps_preserved=$npy;authorization_thread=$base.authorization_thread}
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Archive 'complete.json') -Encoding UTF8
    $result | ConvertTo-Json -Depth 6 -Compress; exit 0
}
if ($Mode -eq 'Verify') {
    if ($stats.files -ne 0) { throw 'Some allowed CCD images remain' }
    $done=Get-Content -LiteralPath (Join-Path $Archive 'complete.json') -Raw | ConvertFrom-Json
    if ($npy.files -ne $done.pixel_memmaps_preserved.files -or $npy.bytes -ne $done.pixel_memmaps_preserved.bytes) { throw 'Protected pixel memmap changed' }
    @{state='verified';remaining_CCD_images=0;pixel_memmaps_preserved=$npy;complete=$done;dependencies=$states} | ConvertTo-Json -Depth 8 -Compress
}
