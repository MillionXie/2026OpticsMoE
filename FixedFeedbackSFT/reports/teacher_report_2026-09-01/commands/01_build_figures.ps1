$ErrorActionPreference = "Stop"

$ReportRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RepoRoot = (Resolve-Path (Join-Path $ReportRoot "..\.." )).Path
$PythonExe = "C:\ProgramData\anaconda3\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Required plotting interpreter not found: $PythonExe"
}

& $PythonExe (Join-Path $ReportRoot "figures\plot_teacher_report.py")
& $PythonExe (Join-Path $RepoRoot ".github\skills\nature-figure\scripts\validate_figure.py") `
    (Join-Path $ReportRoot "figures\plot_teacher_report.py") --backend python --strict
& $PythonExe (Join-Path $RepoRoot ".github\skills\nature-figure\scripts\audit_pdf_text.py") `
    (Join-Path $ReportRoot "figures\fig1_fixed_feedback_evidence.pdf") --min-pt 5
& $PythonExe (Join-Path $RepoRoot ".github\skills\nature-figure\scripts\audit_pdf_text.py") `
    (Join-Path $ReportRoot "figures\fig2_depth_growth_status.pdf") --min-pt 5

Write-Host "Figures rebuilt and validated under $ReportRoot\figures"
