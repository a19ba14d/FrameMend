# 本地 GPU 视频去水印

这套工具使用 Python、FFmpeg 和 NVIDIA GPU 跟踪并修复视频水印。工作流从 JSON 配方读取处理步骤，输出修复后的视频，以及对画面尺寸、时间戳、音频和水印的检查报告。

## 安装

在 Windows 上打开 PowerShell，进入项目目录。需要可通过 `py -3.11` 启动的 Python 3.11，以及支持 CUDA 12.8 和 NVENC 的 NVIDIA GPU 和驱动。

```powershell
.\setup.ps1
```

安装脚本创建 `.venv`，安装 Python 依赖和 CUDA 版 PyTorch，下载并校验 FFmpeg，然后检查 CUDA 和 NVENC。需要 ProPainter 补洞时，安装 Git 并运行 `.\setup.ps1 -ProPainter`。ProPainter 的代码和权重受其 [S-Lab License 1.0](https://github.com/sczhou/ProPainter/blob/main/LICENSE) 约束，商业使用需获得上游许可。

## 用合成视频验证工作流

以下命令在 `runs/` 下生成一秒钟的测试图样和一份只指定画面尺寸的配置，用来验证扫描、渲染和检查流程。

```powershell
$demo = Join-Path 'runs' ('demo-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $demo | Out-Null
$source = Join-Path $demo 'source.mp4'
$recipe = Join-Path $demo 'recipe.json'
$job = Join-Path $demo 'job'
$ffmpeg = (Get-ChildItem -Path '.\.watermark-tools\ffmpeg-*\bin\ffmpeg.exe' -File |
    Select-Object -First 1).FullName
if (!$ffmpeg) { throw 'Run setup.ps1 first.' }

& $ffmpeg -hide_banner -loglevel error -f lavfi -i 'testsrc2=s=320x180:r=30:d=1' -c:v libx264 -pix_fmt yuv420p $source
if ($LASTEXITCODE -ne 0) { throw 'Could not generate the test video.' }
'{"size":[320,180]}' | Set-Content -LiteralPath (Join-Path $demo 'profile.json') -Encoding ascii
'{"profile":"profile.json"}' | Set-Content -LiteralPath $recipe -Encoding ascii

.\restore-workflow.ps1 -Source $source -JobDir $job -Recipe $recipe -CpuDecode
```

运行完成后，`job/` 下会有 `result.mp4`、`run.json`、`verification.json` 和 `watermark-audit.json`。`-CpuDecode` 切换为 CPU 解码，编码仍使用 NVENC。

## 处理自己的视频

- 输入应为恒定帧率的 8 位 SDR `yuv420p` 视频，画面宽高为偶数。跟踪配置的 `size` 必须与输入的宽高一致。
- 配方的 `profile` 指向同目录下的跟踪配置，也可以为阶段指定参数和补洞步骤。先对一个新作业目录传入 `-Plan`，即可预览将执行的阶段。
- 继续中断的任务时，使用相同的源视频、配方和作业目录，并传入 `-Resume`。
- `restore-grok.ps1` 可以将抽取的视频帧发送给兼容的 Grok API，根据回复生成配方。把 `profiles/grok-director.json` 复制为 `profiles/grok-director.local.json` 后，设置 `grok_api_base_url` 和 `model`，通过环境变量 `WM_GROK_API_KEY` 提供凭据，并用 `-Config` 指向本地配置。
- `profiles/local-grok-director.json` 还可连接本地视觉模型。使用这个配置时，按文件中的路径准备自己的模型权重、推理程序和 OCR 依赖。

## 命令与测试

```powershell
.\.venv\Scripts\python.exe watermark.py --help
.\.venv\Scripts\python.exe workflow.py --help
.\.venv\Scripts\python.exe wm_batch.py --help
.\.venv\Scripts\python.exe -m unittest test_workflow_quality test_workflow_recovery
.\.venv\Scripts\python.exe watermark.py self-test
```

## 贡献

本地输入放在 `inputs/`，运行产物放在 `runs/`、`review/` 或 `output/`。提交问题或改动时，请附上可复现的命令、Python 和 FFmpeg 版本以及相关测试结果。可以用上面的合成视频分享复现步骤。

## 许可证

本项目原创源码、配置和文档采用 [Zero-Clause BSD（0BSD）](LICENSE)。允许商用、修改和再分发，无须保留署名或公开修改。

第三方工具、依赖和权重各自适用上游许可。FFmpeg 作为独立工具运行，具体条款取决于 [FFmpeg 构建配置](https://ffmpeg.org/legal.html)。可选的 YOLO 检测和基准测试适用 [Ultralytics 的 AGPL-3.0 或企业授权条款](https://www.ultralytics.com/license)。
