param([switch]$Apply, [string]$ProjectRoot=$PSScriptRoot)
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd('\')
if ((Split-Path $root -Leaf) -ne 'ABO_Lab_SHS_8um') {throw 'Only ABO_Lab_SHS_8um is supported'}
$results=Join-Path $root 'results'
$history=Join-Path $results '90_history'
$reportDir=Join-Path $root 'reports\00_current'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
# Never touch active sessions, SDKs, generated patterns, models, configuration,
# hardware locks, jobs or the current batch. A two-hour cooldown is additional.
$protected=@('90_history','20260913_400us240ms','smoke_configs','smoke_geometry_20260913','smoke_runs','smoke_checks','dual_jobs','phase_guard')
$textFiles=@(Get-ChildItem -LiteralPath $root -File)
foreach ($folder in @('results','reports','generated','compat_abo')) {
  $p=Join-Path $root $folder
  if (Test-Path -LiteralPath $p) {$textFiles+=Get-ChildItem -LiteralPath $p -Recurse -File}
}
$texts=@()
foreach ($f in $textFiles) {
  if ($f.Extension -notin @('.py','.ps1','.json','.yaml','.yml','.md','.html','.txt') -or $f.Length -gt 4MB) {continue}
  if ($f.FullName.StartsWith($history+'\') -or $f.Name -like '04_storage*' -or $f.Name -in @('organize_results.ps1','99_history_index.html')) {continue}
  $texts+=@{path=$f.FullName;text=[IO.File]::ReadAllText($f.FullName)}
}
$rows=@();$moves=@();$cutoff=(Get-Date).AddHours(-2)
foreach ($d in Get-ChildItem -LiteralPath $results -Directory) {
  if ($d.Name -eq '90_history') {continue}
  $files=@(Get-ChildItem -LiteralPath $d.FullName -Recurse -File)
  $refs=@($texts | Where-Object { -not $_.path.StartsWith($d.FullName+'\') -and $_.text.IndexOf($d.Name,[StringComparison]::OrdinalIgnoreCase) -ge 0 } | ForEach-Object { $_.path.Substring($root.Length+1) })
  $recent=@($files | Where-Object {$_.LastWriteTime -gt $cutoff}).Count -gt 0
  # OneDrive regular files carry ReparsePoint as well. Reject actual symbolic
  # links/junctions, not ordinary cloud-backed files within this fixed root.
  $reparse=$d.LinkType -or @($files | Where-Object {$_.LinkType}).Count -gt 0
  $keep=$d.Name -in $protected -or $recent -or $refs.Count -gt 0 -or $reparse
  $row=[ordered]@{name=$d.Name;files=$files.Count;bytes=($files|Measure-Object Length -Sum).Sum;action=$(if($keep){'keep'}else{'archive'});references=$refs;path='results/'+$d.Name}
  if (-not $keep -and $Apply) {
    $source=(Resolve-Path -LiteralPath $d.FullName).Path
    $dest=[IO.Path]::GetFullPath((Join-Path $history $d.Name))
    if ((Split-Path $source -Parent) -ne $results -or -not $dest.StartsWith($history+'\') -or (Test-Path -LiteralPath $dest)) {throw 'Unsafe/duplicate move target'}
    New-Item -ItemType Directory -Force -Path $history | Out-Null
    Move-Item -LiteralPath $source -Destination $dest
    $moves+=@{source=$source;destination=$dest;files=$files.Count;bytes=$row.bytes}
    $row.path='results/90_history/'+$d.Name
  }
  $rows+=[pscustomobject]$row
}
$mode=if($Apply){'applied'}else{'preview'}
$journal=[ordered]@{time=(Get-Date).ToString('s');mode=$mode;root=$root;deleted_files=0;freed_bytes=0;moves=$moves;directories=$rows}
$file=Join-Path $reportDir $(if($Apply){'04_storage_applied_'+(Get-Date -Format yyyyMMdd_HHmmss)+'.json'}else{'04_storage_preview.json'})
$journal | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $file -Encoding UTF8
$page=@('<!doctype html><meta charset="utf-8"><title>ABO files</title><style>body{font:17px/1.6 sans-serif;max-width:1300px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:7px}code{word-break:break-all}</style>',
 '<h1>ABO 工程文件导航</h1><p><a href="03_live.html">当前逐层采集</a> · <a href="../../COMMAND.md">实验指令</a> · <a href="../../generated/">标定/相位BMP</a> · <a href="../../sessions/">真实CCD会话（师弟电脑）</a></p>',
 '<p>历史诊断保留证据，不作为当前推荐参数。归档只移动没有发现外部文本引用、且至少两小时未写入的目录；当前批次/会话/标定不动。未删除CCD或权重。</p><table><tr><th>用途</th><th>目录</th><th>MiB</th><th>状态</th></tr>')
foreach ($r in $rows | Sort-Object action,name) {
  $category=switch -Regex ($r.name) {'mnist' {'MNIST对照';break} 'timing|digits|candidate|gray|exposure' {'曝光/时序';break} 'smoke|400us' {'六层实验/配置';break} 'phase|origin|alignment|check|marker' {'相位/对齐诊断';break} default {'相机/早期调试'}}
  $name=[Net.WebUtility]::HtmlEncode($r.name);$path=[Net.WebUtility]::HtmlEncode($r.path)
  $page+='<tr><td>'+ $category+'</td><td><a href="../../'+$path+'/">'+$name+'</a></td><td>'+[math]::Round($r.bytes/1MB,2)+'</td><td>'+$r.action+'</td></tr>'
}
$page+='</table><p>恢复：按同目录04_storage_applied日志的source/destination，将指定历史目录移回原位置即可；不要覆盖同名新实验。</p>'
$page -join "`n" | Set-Content -LiteralPath (Join-Path $reportDir '04_storage.html') -Encoding UTF8
if ($Apply -and (Test-Path -LiteralPath (Join-Path $reportDir '99_history_index.html'))) {
  '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=04_storage.html"><a href="04_storage.html">历史结果已统一整理，打开分类目录</a>' | Set-Content -LiteralPath (Join-Path $reportDir '99_history_index.html') -Encoding UTF8
}
Write-Output "Mode=$mode directories=$($rows.Count) candidates=$(@($rows | Where-Object action -eq 'archive').Count) moved=$($moves.Count)"
Write-Output "Journal=$file"
$rows | Where-Object action -eq 'archive' | Select-Object name,files,@{Name='MiB';Expression={[math]::Round($_.bytes/1MB,2)}} | Format-Table -AutoSize
