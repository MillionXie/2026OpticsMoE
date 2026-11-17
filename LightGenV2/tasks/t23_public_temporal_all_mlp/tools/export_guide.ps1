param(
    [Parameter(Mandatory = $true)][string]$DocumentPath,
    [Parameter(Mandatory = $true)][string]$PdfPath
)

$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $DocumentPath).Path
$outputPath = [System.IO.Path]::GetFullPath($PdfPath)
New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$wordApplication = New-Object -ComObject Word.Application
$ownedApplication = $false
$wordDocument = $null
try {
    if ($wordApplication.Documents.Count -ne 0) { throw 'The created Word instance contains another document' }
    $ownedApplication = $true
    $wordApplication.Visible = $false
    $wordApplication.DisplayAlerts = 0
    $wordDocument = $wordApplication.Documents.Open($sourcePath, $false, $true)
    $wordDocument.Fields.Update() | Out-Null
    foreach ($section in $wordDocument.Sections) {
        foreach ($footer in $section.Footers) { $footer.Range.Fields.Update() | Out-Null }
    }
    $wordDocument.ExportAsFixedFormat($outputPath, 17)
    Write-Output ('Rendered ' + $wordDocument.ComputeStatistics(2) + ' pages')
}
finally {
    if ($null -ne $wordDocument) { $wordDocument.Close(0) }
    if ($ownedApplication) { $wordApplication.Quit(0) }
}
