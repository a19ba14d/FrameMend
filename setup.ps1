param([switch]$ProPainter)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program exited with code $LASTEXITCODE" }
}

function Get-VerifiedFile {
    param([string]$Url, [string]$Destination, [string]$Sha256)
    if (Test-Path -LiteralPath $Destination) {
        if ((Get-FileHash -LiteralPath $Destination).Hash -ne $Sha256) {
            throw "SHA256 mismatch for $Destination"
        }
        return
    }
    $partial = "$Destination.$PID.part"
    try {
        Invoke-Checked 'curl.exe' @('--fail', '--location', '--retry', '2', '--silent', '--show-error', $Url, '--output', $partial)
        if ((Get-FileHash -LiteralPath $partial).Hash -ne $Sha256) { throw "SHA256 mismatch for $Url" }
        Move-Item -LiteralPath $partial -Destination $Destination
    }
    finally {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial }
    }
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (!(Test-Path -LiteralPath $venvPython)) {
        Invoke-Checked 'py.exe' @('-3.11', '-m', 'venv', '.venv')
    }
    Invoke-Checked $venvPython @('-m', 'pip', 'install', '-r', 'requirements.txt')
    Invoke-Checked $venvPython @('-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu128', 'torch==2.11.0+cu128', 'torchvision==0.26.0+cu128')
    Invoke-Checked $venvPython @('-c', 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), (torch.ones((32,32), device="cuda") @ torch.ones((32,32), device="cuda"))[0,0].item())')

    $toolsDir = Join-Path $PSScriptRoot '.watermark-tools'
    New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
    $ffmpeg = Join-Path $toolsDir 'ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe'
    $ffprobe = Join-Path $toolsDir 'ffmpeg-9.0.2-essentials_build\bin\ffprobe.exe'
    if (!(Test-Path -LiteralPath $ffmpeg) -or !(Test-Path -LiteralPath $ffprobe)) {
        $zipPath = Join-Path $toolsDir 'ffmpeg-9.0.2.zip'
        Get-VerifiedFile 'https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip' $zipPath '60F467265B1E312373DBCD92200C2618A74850F98D3D078E94296BB3FA2047BA'
        Invoke-Checked 'tar.exe' @('-xf', $zipPath, '-C', $toolsDir)
    }
    Invoke-Checked $ffmpeg @('-hide_banner','-v','error','-f','lavfi','-i','testsrc2=s=320x180:r=30:d=0.2','-c:v','h264_nvenc','-f','null','NUL')

    if ($ProPainter) {
        $repo = Join-Path $toolsDir 'ProPainter'
        $commit = '326782529a63d997cab80ab21439b92ed422b3d3'
        if (!(Test-Path -LiteralPath $repo)) {
            Invoke-Checked 'git.exe' @('init', $repo)
            Invoke-Checked 'git.exe' @('-C', $repo, 'fetch', '--depth=1', 'https://github.com/sczhou/ProPainter.git', $commit)
            Invoke-Checked 'git.exe' @('-C', $repo, 'checkout', '--detach', 'FETCH_HEAD')
        }
        $checkedOut = & git.exe -C $repo rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or $checkedOut -ne $commit) { throw "Expected ProPainter revision $commit in $repo" }
        $weightsDir = Join-Path $repo 'weights'
        $weights = @{
            'ProPainter.pth' = '12C070C4B48F374C91D8A2A17851140B85C159621080989F9E191BBC18BD6591'
            'raft-things.pth' = 'FCFA4125D6418F4DE95D84AEC20A3C5F4E205101715A79F193243C186AC9A7E1'
            'recurrent_flow_completion.pth' = '22939A1A7900DA878DBE1CCD011D646B1BFB30B8290039D8FF0E0C2FEFBFD283'
        }
        foreach ($name in $weights.Keys) {
            Get-VerifiedFile "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/$name" (Join-Path $weightsDir $name) $weights[$name]
        }
    }
    Write-Output 'Local CUDA and NVENC are ready.'
}
finally {
    Pop-Location
}
