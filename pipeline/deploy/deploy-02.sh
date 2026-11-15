#!/usr/bin/env bash
# From a clean checkout: pipeline/deploy/deploy-02.sh [ref]   (default HEAD; production uses main)
# Streams `git archive` to 02 over ssh and runs install.sh there. Needs the "02" host in ~/.ssh/config.
set -euo pipefail
ref=${1:-HEAD}
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain) ]] || { echo "working tree not clean" >&2; exit 1; }
sha=$(git rev-parse --short=12 "$ref^{commit}")
staging=/srv/framemend/incoming/$sha
{ git archive --format=tar "$ref"; } | ssh 02 "wsl.exe -d Ubuntu-24.04 -u root -- bash -c 'set -e; rm -rf $staging; mkdir -p $staging; tar -x -C $staging; echo $sha > $staging/VERSION'"
ssh 02 "wsl.exe -d Ubuntu-24.04 -u root -- bash $staging/pipeline/deploy/install.sh $staging"
