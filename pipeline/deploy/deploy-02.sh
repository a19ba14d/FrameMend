#!/usr/bin/env bash
# From a clean checkout: pipeline/deploy/deploy-02.sh [ref]   (default HEAD; production uses main)
# Streams `git archive` to 02 over ssh and runs install.sh there. Needs the "02" host in ~/.ssh/config.
# 02 logs into Windows cmd, so every remote command runs through wsl.exe without shell quoting.
set -euo pipefail
ref=${1:-HEAD}
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain) ]] || { echo "working tree not clean" >&2; exit 1; }
sha=$(git rev-parse --short=12 "$ref^{commit}")
staging=/srv/framemend/incoming/$sha
wsl() { ssh 02 "wsl.exe -d Ubuntu-24.04 -u root -- $*"; }
wsl rm -rf "$staging"
wsl mkdir -p "$staging"
git archive --format=tar "$ref" | wsl tar -x -C "$staging"
printf 'printf "%%s\\n" %s > %s/VERSION\n' "$sha" "$staging" | wsl bash -s
wsl test -f "$staging/VERSION"
wsl bash "$staging/pipeline/deploy/install.sh" "$staging"
