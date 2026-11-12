param(
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$JobDir,
    [string]$Config,
    [string]$Recipe,
    [string]$ReusePrepared,
    [switch]$DetectKnownLayout,
    [switch]$Recheck,
    [switch]$CpuDecode
)

$ErrorActionPreference = 'Stop'
if (!$Config) { $Config = Join-Path $PSScriptRoot 'profiles/grok-director.json' }
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$entry = Join-Path $PSScriptRoot 'wm_director.py'
$options = @()
if ($Recipe) { $options += @('--recipe', $Recipe) }
if ($ReusePrepared) { $options += @('--reuse-prepared', $ReusePrepared) }
if ($DetectKnownLayout) { $options += @('--detect-known-layout') }
if ($Recheck) { $options += '--recheck' }
if ($CpuDecode) { $options += '--cpu-decode' }
& $python -X utf8 $entry $Source $JobDir --config $Config @options
exit $LASTEXITCODE
