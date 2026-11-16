# FrameMend

FrameMend 是一套本地 GPU 视频去水印工具，使用 Python、FFmpeg 和 NVIDIA GPU 跟踪并修复视频水印。工作流从 JSON 配方读取处理步骤，输出修复后的视频，以及对画面尺寸、时间戳、音频和水印的检查报告。

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

## 自动去水印（无需配方）

`wm_auto.py` 自己找出视频里反复出现的水印文字（角标、移动水印、整幅半透明叠加），学习外形、逐帧跟踪，生成配方后修补，再用 OCR 复查结果；仍有残留时会加大掩码重试。整幅半透明叠加面积太大、无法修补时，按多帧平均画面估计透明度后做反向混合。

移动水印划过烧录字幕时，修补会把被盖住的字一起抹掉。修补后，`wm_auto.py` 用源视频和修补结果拟合这个水印的透明度和颜色，在字幕行内对源画面做反向混合，把白字黑边的字形取回来，写成 `restored.mp4`，作业的 `result.mp4` 指向它。只有和本帧可见字幕连成一行、或者自身够一个完整字的笔画才会取回，画面里零散的高光不会被当成字。加 `--no-captions` 关闭这一步。

Linux 或 WSL2（Ubuntu）上用 `setup.sh` 安装，不需要 sudo：

```bash
./setup.sh --propainter
```

Python 环境默认建在 `~/.venvs/framemend`，可用环境变量 `WATERMARK_VENV` 改到别处。在 WSL2 中建议放在 Linux 文件系统里，导入会快很多。

处理单个视频（作业目录必须是新的）。加 `--plan-only` 只生成计划和预览图，不修补；加 `--cpu-decode` 改用 CPU 解码；加 `--audit` 在每次尝试后额外做一遍逐像素审计（默认不做，验收只看 OCR 复查）：

```bash
~/.venvs/framemend/bin/python wm_auto.py input.mp4 runs/job-1
```

处理整个目录。中断后用同样的命令继续：已完成的视频会跳过，中断的重新开始；加 `--retry-failed` 重试失败的视频。多块 GPU 时，每块 GPU 启动一个进程并指向同一个作业目录，进程之间按文件加锁分配视频：

```bash
CUDA_VISIBLE_DEVICES=0 ~/.venvs/framemend/bin/python wm_auto.py inputs/ runs/batch &
CUDA_VISIBLE_DEVICES=1 ~/.venvs/framemend/bin/python wm_auto.py inputs/ runs/batch &
```

每个视频的结果在 `runs/batch/<视频名>/`：`result.mp4`、`auto.json`（状态、原因、各次尝试）和 `events.jsonl`（逐步记录）。`runs/batch/batch.json` 汇总所有视频。状态含义：

- `complete`：水印已去除，OCR 复查通过。
- `no_watermark`：没有找到反复出现的水印文字。
- `needs_review`：已输出结果，但需要人工确认，`reason` 写明原因，例如半透明叠加只被减弱、可能仍有残影。
- `failed`：处理出错，`error` 写明原因。

## 接入 02 投稿链路（高热剧去水印）

`pipeline/framemend_pipeline.py` 把 `wm_auto.py` 接进 02 上的爬虫和成品投稿（produced-submit），按「去水印热度标准」只处理高热剧：

- dupeguard 每 15 分钟给高热且未投稿的剧写 `.dewatermark`。`framemend-queue.service` 按热度（热度值 / 门槛）从高到低逐集跑 `wm_auto.py`（已经由 produced-submit 发布过的集跳过：dupeguard 投稿后不撤标记；小于该来源投稿门槛 `source_min_size` 的集也跳过，它们本来就不会投），成品放在 `/srv/framemend/out/`，每集的状态在 `/srv/framemend/state/`。不改爬虫目录。
- `/srv/framemend/bin/framemend-guard` 作为 `PRODUCED_WORKER`，包在真正的 produced-worker 外面：
  - 剧没有标记，或来源不在 `enabled_sources` 里：参数原样交给 worker。
  - 高热剧的集还没去完水印，或在等人工确认：返回 75。produced-submit 记一条 FAIL，下一轮用同样的参数重试，所以高热剧会一直等到去完水印再投。
  - 已完成：把 `--source` 换成去水印成品后交给 worker。
  - 未发现水印、人工放行：投原片。
  - 投稿会话已经用原片开始了：保持原片，因为 worker 拒绝换文件。
- worker 发布去水印成品后，队列服务每 5 分钟读一次 worker 的回执，调用中台 `POST /api/v1/produced-submissions/{id}/dewatermark`。中台把视频记为「已去水印」，后台内容库显示徽标，Site API 返回 `dewatermarked`。标记成功 2 天后删除成品。原片仍由 `release-published.py` 按回执清理。
- `needs_review`（OCR 复查仍认出文字，常见是把画面内容误认成字母）直接进入「待人工」，不再重跑：成品留在 `/srv/framemend/review/`，残留文字所在帧框好红框截图放在旁边，`framemend-ctl status` 会列出来。人看过没问题就 `accept`，投去水印成品。
- 判成 `complete` 但所有水印层加起来只出现在不到一半的帧里（`min_coverage`，默认 0.5），也进「待人工」：真水印几乎每帧都在，这种多半是把停留几秒的字幕当成了静态水印、把字幕抹掉了。看截图后，误判用 `release` 投原片，确有水印用 `accept`。
- `failed` 会加大掩码（`--dilation 8`）重试 1 次，仍失败就进入「待人工」。待人工的集继续压着不投。

部署（在干净检出上，生产用 `main`）：`pipeline/deploy/deploy-02.sh main`。它会把代码装到 `/srv/framemend/app`，第一次部署时建 GPU 环境 `/srv/framemend/venv`，并装好 `framemend-queue.service` 和 `framemend-ctl`，但不会自动启用。配置文件是 `/srv/framemend/config.json`，样例见 `pipeline/deploy/config.example.json`；`enabled_sources` 决定哪些来源被拦下等待去水印。

给某个来源接入时，在 `/srv/produced/bin/produced-<源>-local.sh` 最后的 `exec` 前加下面这段，改之前先把原文件备份到 `/srv/produced/ops/framemend-on-<日期>/`：

```bash
# framemend-entry-v1
export FRAMEMEND_REAL_WORKER="${PRODUCED_WORKER:?}"
export PRODUCED_WORKER=/srv/framemend/bin/framemend-guard
```

运维命令：

```bash
framemend-ctl status                    # 各来源的状态计数、排队集数、平均耗时、预计积压小时数、待人工清单、标记情况
framemend-ctl pause | resume            # 暂停 / 恢复队列（已完成的照常投）
framemend-ctl accept lizhi 530 [3]      # 待人工的集看过截图确认干净，投去水印成品
framemend-ctl release lizhi 530 [3]     # 人工确认后让待处理的集改投原片（带水印）
framemend-ctl retry lizhi 530 [3]       # 待人工的集加大掩码重新排队
```

多一台 GPU 机（例如 dell）：

1. 先用 `FRAMEMEND_HOST=dell pipeline/deploy/deploy-02.sh main` 把同一版本装到那台机器。它的队列单元不要启用。
2. 在 02 的 `config.json` 里加 `"remote_workers": {"dell": {"ssh": ["ssh", "-o", "BatchMode=yes", "-i", "/root/.ssh/framemend_dell", "dell@<IP>"]}}`。
3. 启用 `framemend-queue-remote@dell`。

工作方式：

- 这个进程跑在 02 上，和本机队列共用候选列表，靠每集一把锁分开，各处理不同的集。
- 每集的流程是：原片经 ssh 传到对方的 `/srv/framemend/remote/`，在对方跑 `wm_auto`，成品和报告传回 02。两个方向都校验 sha256，对方机器上不留文件。状态、复核、标记仍然都在 02 上。
- ssh 或传输出错时，这一集不算失败、不消耗重试次数，该进程过 5 分钟再试。

回滚：恢复 `produced-<源>-local.sh` 的备份，或者把该来源从 `enabled_sources` 里去掉（guard 会立即放行），然后 `systemctl disable --now framemend-queue`。

## 命令与测试

```powershell
.\.venv\Scripts\python.exe watermark.py --help
.\.venv\Scripts\python.exe workflow.py --help
.\.venv\Scripts\python.exe wm_batch.py --help
.\.venv\Scripts\python.exe -m unittest test_workflow_quality test_workflow_recovery test_pipeline
.\.venv\Scripts\python.exe watermark.py self-test
```

## 贡献

本地输入放在 `inputs/`，运行产物放在 `runs/`、`review/` 或 `output/`。提交问题或改动时，请附上可复现的命令、Python 和 FFmpeg 版本以及相关测试结果。可以用上面的合成视频分享复现步骤。

## 许可证

本项目原创源码、配置和文档采用 [Zero-Clause BSD（0BSD）](LICENSE)。允许商用、修改和再分发，无须保留署名或公开修改。

第三方工具、依赖和权重各自适用上游许可。FFmpeg 作为独立工具运行，具体条款取决于 [FFmpeg 构建配置](https://ffmpeg.org/legal.html)。可选的 YOLO 检测和基准测试适用 [Ultralytics 的 AGPL-3.0 或企业授权条款](https://www.ultralytics.com/license)。
