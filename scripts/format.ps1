<#
.SYNOPSIS
    Formats project sources under src/ with clang-format. Never touches third_party/.

.PARAMETER Check
    Change nothing; exit with a non-zero code if any file is not formatted.

.EXAMPLE
    .\scripts\format.ps1
    .\scripts\format.ps1 -Check

.NOTES
    Override the binary with $env:CLANG_FORMAT. clang-format ships with the
    "C++ Clang tools for Windows" component of Visual Studio 2022.
#>
[CmdletBinding()]
param(
    [switch]$Check
)

$ErrorActionPreference = 'Stop'

$clangFormat = if ($env:CLANG_FORMAT) { $env:CLANG_FORMAT } else { 'clang-format' }
if (-not (Get-Command $clangFormat -ErrorAction SilentlyContinue)) {
    Write-Error "format.ps1: '$clangFormat' not found in PATH"
    exit 2
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$srcDir = Join-Path $repoRoot 'src'

$files = @(Get-ChildItem -Path $srcDir -Recurse -File |
    Where-Object { $_.Extension -in '.cpp', '.hpp', '.h' } |
    Sort-Object FullName |
    ForEach-Object { $_.FullName })

if ($files.Count -eq 0) {
    Write-Host 'format.ps1: no source files found under src/'
    exit 0
}

Push-Location $repoRoot
try {
    if ($Check) {
        & $clangFormat --style=file --dry-run --Werror @files
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'format.ps1: some files are not formatted, run scripts/format.ps1'
            exit 1
        }
        Write-Host "format.ps1: $($files.Count) file(s) checked, all formatted"
    }
    else {
        & $clangFormat --style=file -i @files
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        Write-Host "format.ps1: $($files.Count) file(s) formatted"
    }
}
finally {
    Pop-Location
}
