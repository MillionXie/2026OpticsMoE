$ErrorActionPreference='Stop'
$taskRoot=Join-Path $PSScriptRoot 'outputs/01a061e2-6f88-7913-87f5-b632000e4c83/collected_20261025_01'
$taskInput=Join-Path $taskRoot 'A100_时间能耗与倍数计算说明_20261025.docx'
$taskQa=Join-Path $taskRoot 'docx_qa'
New-Item -ItemType Directory -Path $taskQa -Force | Out-Null
$taskPdf=Join-Path $taskQa 'methods_native_word.pdf'
$taskWord=$null
$taskDocument=$null
try {
  $taskWord=New-Object -ComObject Word.Application
  if ($taskWord.Documents.Count -ne 0) { throw 'Unexpected pre-existing documents; do not touch user Word state' }
  $taskWord.Visible=$false
  $taskWord.DisplayAlerts=0
  $taskWord.AutomationSecurity=3
  $taskDocument=$taskWord.Documents.Open($taskInput,$false,$true,$false)
  $taskDocument.ExportAsFixedFormat($taskPdf,17)
  $taskPages=$taskDocument.ComputeStatistics(2)
  $taskDocument.Close(0)
  [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskDocument) | Out-Null
  $taskDocument=$null
  $taskWord.Quit(0)
  [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskWord) | Out-Null
  $taskWord=$null
  [pscustomobject]@{pdf=$taskPdf;pages=$taskPages} | ConvertTo-Json -Compress
} finally {
  if ($null -ne $taskDocument) { $taskDocument.Close(0); [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskDocument) | Out-Null }
  if ($null -ne $taskWord -and $taskWord.Documents.Count -eq 0) { $taskWord.Quit(0); [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($taskWord) | Out-Null }
}
