param([Parameter(Mandatory=$true)][string]$Document,[Parameter(Mandatory=$true)][string]$OutputDirectory)
$taskDoc=(Resolve-Path -LiteralPath $Document).Path
$taskRoot=(Resolve-Path -LiteralPath $PSScriptRoot).Path
if(-not $taskDoc.StartsWith($taskRoot+[IO.Path]::DirectorySeparatorChar)){throw 'Document must remain within protocol directory'}
[void](New-Item -ItemType Directory -Force -Path $OutputDirectory)
$taskOut=(Resolve-Path -LiteralPath $OutputDirectory).Path
if(-not $taskOut.StartsWith($taskRoot+[IO.Path]::DirectorySeparatorChar)){throw 'QA output must remain within protocol directory'}
$taskPDF=Join-Path $taskOut 'methods.pdf'
$taskWord=$null
$taskOpened=$null
try {
    $taskWord=New-Object -ComObject Word.Application
    $taskWord.Visible=$false
    $taskWord.DisplayAlerts=0
    $taskOpened=$taskWord.Documents.Open($taskDoc,$false,$true)
    $taskOpened.ExportAsFixedFormat($taskPDF,17)
    Write-Output $taskPDF
} finally {
    if($null -ne $taskOpened){$taskOpened.Close(0)}
    if($null -ne $taskWord){$taskWord.Quit()}
}
