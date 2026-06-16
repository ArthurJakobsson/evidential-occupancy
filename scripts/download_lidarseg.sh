#!/usr/bin/env bash
# Fetch + install the nuScenes lidarseg expansion under $NUSCENES_ROOT.
#
# nuScenes lidarseg is gated behind the nuScenes Terms of Use, so there is no public
# direct/gdown link and it CANNOT be fetched non-interactively. You provide the source:
#
#   1) Signed URL (recommended): log in at https://www.nuscenes.org/nuscenes#download,
#      accept the terms, right-click the "nuScenes-lidarseg-all (v1.0)" link -> Copy link,
#      then:
#        source paths.env
#        bash scripts/download_lidarseg.sh "<paste-signed-url>"
#
#   2) A Google Drive mirror you control (a personal copy — do NOT use third-party
#      redistributions; nuScenes forbids redistribution):
#        source paths.env
#        bash scripts/download_lidarseg.sh --gdown <drive-file-id>
#
# The tarball extracts to add, under $NUSCENES_ROOT:
#   lidarseg/v1.0-{mini,trainval,test}/*.bin
#   v1.0-{mini,trainval}/lidarseg.json
#   v1.0-{mini,trainval}/category.json   (now with the `index` field)
# After it lands, the lidarseg-transfer stage can run (set with_lidarseg=true is handled).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${NUSCENES_ROOT:?set NUSCENES_ROOT (e.g. 'source paths.env') first}"
PIXI="${PIXI:-pixi}"
TARBALL="${TMPDIR:-/tmp}/nuScenes-lidarseg-all-v1.0.tar.bz2"

if [ "${1:-}" = "--gdown" ]; then
    [ -n "${2:-}" ] || { echo "usage: $0 --gdown <drive-file-id>"; exit 1; }
    "$PIXI" run python -c "import gdown" 2>/dev/null || "$PIXI" run pip install gdown
    "$PIXI" run gdown "$2" -O "$TARBALL"
elif [ -n "${1:-}" ]; then
    echo "downloading from signed URL ..."
    curl -L --fail -o "$TARBALL" "$1"
else
    echo "usage: $0 <signed-url>   |   $0 --gdown <drive-file-id>"
    exit 1
fi

echo "extracting into $NUSCENES_ROOT ..."
tar -xjf "$TARBALL" -C "$NUSCENES_ROOT"
echo "done. verify:"
echo "  ls $NUSCENES_ROOT/lidarseg/v1.0-mini | head"
ls "$NUSCENES_ROOT"/lidarseg/v1.0-mini 2>/dev/null | head -3 || echo "  (lidarseg/v1.0-mini not found — check the tarball layout)"
