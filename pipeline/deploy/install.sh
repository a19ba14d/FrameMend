#!/usr/bin/env bash
# Runs on 02 (WSL, root) from an unpacked release directory: install.sh <release dir>
# Installs code to /srv/framemend/app, builds the GPU venv on first run, installs units (not started).
set -euo pipefail
release=$(cd "$1" && pwd)
root=/srv/framemend
mkdir -p "$root"/{bin,state,jobs,out,bindings,locks,releases}
chmod 700 "$root"
version=$(cat "$release/VERSION")
target=$root/releases/$version
if [[ ! -d $target ]]; then
    cp -a "$release" "$target.tmp" && mv "$target.tmp" "$target"
fi
# Share downloaded tools (ffmpeg, uv, ProPainter weights) across releases.
mkdir -p "$root/tools"
ln -sfn "$root/tools" "$target/.watermark-tools"
ln -sfn "$target" "$root/app.new" && mv -T "$root/app.new" "$root/app"
WATERMARK_VENV=$root/venv WATERMARK_TOOLS_DIR=$root/tools "$root/app/setup.sh" --propainter
install -m 755 "$root/app/pipeline/deploy/framemend-guard" "$root/bin/framemend-guard"
install -m 755 "$root/app/pipeline/deploy/framemend-ctl" /usr/local/bin/framemend-ctl
install -m 644 "$root/app/pipeline/deploy/framemend-queue.service" /etc/systemd/system/framemend-queue.service
[[ -f $root/config.json ]] || install -m 600 "$root/app/pipeline/deploy/config.example.json" "$root/config.json"
systemctl daemon-reload
echo "FrameMend $version installed at $root/app; queue unit installed (enable with: systemctl enable --now framemend-queue)"
