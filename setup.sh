#!/usr/bin/env bash
# Linux / WSL2 counterpart of setup.ps1. No sudo needed.
#
#   ./setup.sh [--propainter]
#
# Creates a Python 3.11 environment with CUDA torch and local OCR (default ~/.venvs/framemend, override with
# WATERMARK_VENV; a Linux file system keeps imports fast under WSL), and unpacks a static ffmpeg with
# NVDEC/NVENC under .watermark-tools, where wm_media finds it.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=${WATERMARK_TOOLS_DIR:-$ROOT/.watermark-tools}
VENV=${WATERMARK_VENV:-$HOME/.venvs/framemend}
UV_VERSION=0.13.1
UV_SHA256=eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220
# BtbN prunes old autobuilds; point FFMPEG_URL/FFMPEG_SHA256 at a current n9.0 linux64-gpl build when this one is gone.
FFMPEG_NAME=ffmpeg-n9.0.2-49-gc179411bbb-linux64-gpl-9.0
FFMPEG_URL=${FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-11-11-12-39/$FFMPEG_NAME.tar.xz}
FFMPEG_SHA256=${FFMPEG_SHA256:-ca1107e3a14a9de4c4712f9405bb720a47dac41426cf2a1da09acb4811805758}
PROPAINTER_COMMIT=326782529a63d997cab80ab21439b92ed422b3d3

propainter=0
for argument in "$@"; do
    case $argument in
        --propainter) propainter=1 ;;
        *) echo "Unknown argument $argument" >&2; exit 2 ;;
    esac
done

fetch() {  # url destination sha256
    local url=$1 destination=$2 sha=$3
    if [[ -f $destination ]]; then
        echo "$sha  $destination" | sha256sum -c --quiet - || { echo "SHA256 mismatch for $destination" >&2; exit 1; }
        return
    fi
    local partial="$destination.$$.part"
    if ! curl --fail --location --retry 2 --silent --show-error "$url" --output "$partial" \
        || ! echo "$sha  $partial" | sha256sum -c --quiet -; then
        rm -f "$partial"
        echo "Download or SHA256 check failed for $url" >&2
        exit 1
    fi
    mv "$partial" "$destination"
}

mkdir -p "$TOOLS"

uv="$TOOLS/uv-$UV_VERSION/uv"
if [[ ! -x $uv ]]; then
    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
        "$TOOLS/uv-$UV_VERSION.tar.gz" "$UV_SHA256"
    mkdir -p "$TOOLS/uv-$UV_VERSION"
    tar -xzf "$TOOLS/uv-$UV_VERSION.tar.gz" -C "$TOOLS/uv-$UV_VERSION" --strip-components=1
fi

[[ -x $VENV/bin/python ]] || "$uv" venv --python 3.11 "$VENV"
python=$VENV/bin/python
"$uv" pip install --python "$python" -r "$ROOT/requirements.txt"
"$uv" pip install --python "$python" --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128
"$uv" pip install --python "$python" rapidocr==3.10.0 onnxruntime==1.31.1 pyclipper==1.4.0 shapely==2.2.0

# Reuse OCR models another setup already fetched; RapidOCR downloads whatever is still missing on first use.
models=$("$python" -I -c 'import pathlib, rapidocr; print(pathlib.Path(rapidocr.__file__).parent/"models")')
mkdir -p "$models"
for model in "$TOOLS"/vision/ocr/rapidocr/models/*.onnx; do
    if [[ -f $model && ! -f $models/$(basename "$model") ]]; then
        cp "$model" "$models/"
    fi
done
"$python" -I -c 'from rapidocr import RapidOCR; RapidOCR(params={"Global.use_cls": False, "Global.log_level": "error"})'

"$python" -I -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), (torch.ones((32,32), device="cuda") @ torch.ones((32,32), device="cuda"))[0,0].item())'

ffmpeg_dir=$TOOLS/$FFMPEG_NAME
if [[ ! -x $ffmpeg_dir/bin/ffmpeg || ! -x $ffmpeg_dir/bin/ffprobe ]]; then
    fetch "$FFMPEG_URL" "$TOOLS/$FFMPEG_NAME.tar.xz" "$FFMPEG_SHA256"
    tar -xJf "$TOOLS/$FFMPEG_NAME.tar.xz" -C "$TOOLS"
fi
# The workflow decodes with NVDEC + scale_cuda and encodes with NVENC; check both ends.
probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
"$ffmpeg_dir/bin/ffmpeg" -hide_banner -v error -f lavfi -i testsrc2=s=320x180:r=30:d=0.2 -c:v h264_nvenc "$probe/probe.mp4"
"$ffmpeg_dir/bin/ffmpeg" -hide_banner -v error -hwaccel cuda -hwaccel_output_format cuda -i "$probe/probe.mp4" \
    -vf scale_cuda=160:90,hwdownload,format=nv12 -f null -

if (( propainter )); then
    repo=$TOOLS/ProPainter
    if [[ ! -d $repo ]]; then
        git init "$repo"
        git -C "$repo" fetch --depth=1 https://github.com/sczhou/ProPainter.git "$PROPAINTER_COMMIT"
        git -C "$repo" checkout --detach FETCH_HEAD
    fi
    [[ $(git -C "$repo" rev-parse HEAD) == "$PROPAINTER_COMMIT" ]] || { echo "Expected ProPainter revision $PROPAINTER_COMMIT in $repo" >&2; exit 1; }
    mkdir -p "$repo/weights"
    while read -r name sha; do
        fetch "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/$name" "$repo/weights/$name" "$sha"
    done <<'EOF'
ProPainter.pth 12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591
raft-things.pth fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1
recurrent_flow_completion.pth 22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283
EOF
fi

echo "Local CUDA, OCR and NVENC are ready. Run the workflow with $python"
