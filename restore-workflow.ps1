param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$JobDir,
    [Parameter(Mandatory=$true)][string]$Recipe,
    [switch]$CpuDecode,
    [switch]$Plan,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$entry = Join-Path $PSScriptRoot 'workflow.py'
$options = @()
if ($CpuDecode) { $options += '--cpu-decode' }
if ($Plan) { $options += '--plan' }
if ($Resume) { $options += '--resume' }
& $python -X utf8 $entry $Source $JobDir --recipe $Recipe @options
if ($LASTEXITCODE -ne 0) { throw 'Workflow stopped. Check the output and run.json in the job directory.' }
