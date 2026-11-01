#!/usr/bin/env bash
# Install the topo map pipeline:
#   1. Python package (editable, console entry point: topo-map-generator)
#   2. EyePiece binary (built by scripts/build-eyepiece.sh) onto PATH
#
# Usage:
#   scripts/install.sh [bin-dir]      # default: ~/.local/bin
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${1:-$HOME/.local/bin}"
BUILD="${BUILD_DIR:-$ROOT/osmand-native/build}"

echo ">> [1/2] Python package (editable)"
if [ -d "$ROOT/.venv" ]; then
    PIP="$ROOT/.venv/bin/pip"
else
    PIP="pip3"
fi
"$PIP" install -e "$ROOT"

echo ">> [2/2] EyePiece binary -> $BIN_DIR"
EYEPIECE="$BUILD/tools/cpp-tools/map-rasterizer/eyepiece"
if [ ! -x "$EYEPIECE" ]; then
    echo "EyePiece not built yet - run scripts/build-eyepiece.sh first" >&2
    exit 1
fi
mkdir -p "$BIN_DIR"
install -m 755 "$EYEPIECE" "$BIN_DIR/eyepiece"

echo
echo "Done. Verify with:"
echo "  topo-map-generator --help"
echo "  eyepiece -help"
echo
echo "Data note: OBF files / GeoTIFF heightmaps live outside the repo"
echo "(default --obf_dir ~/osmand-data; EyePiece expects <obf_dir>/geotiff)."
