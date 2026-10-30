#!/usr/bin/env bash
# Build the EyePiece rasterizer (OsmAnd core + tools, topo-map patches).
#
# Usage:
#   scripts/build-eyepiece.sh              # configure + build
#   JOBS=8 BUILD=/path/to/build scripts/build-eyepiece.sh
#
# Requirements (Debian/Ubuntu): build-essential cmake ninja-build git
#   libgdal-dev libgl1-mesa-dev libglew-dev (core bundles the rest in
#   osmand-native/core/externals)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/osmand-native"
BUILD="${BUILD_DIR:-$SRC/build}"
JOBS="${JOBS:-$(nproc)}"
TARGET="${TARGET:-eyepiece}"

if [ ! -d "$SRC/core" ]; then
    echo "osmand-native submodule missing. Run: git submodule update --init" >&2
    exit 1
fi

if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    echo ">> Configuring ($BUILD)"
    cmake -S "$SRC" -B "$BUILD" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENGL_egl_LIBRARY=/usr/lib/x86_64-linux-gnu/libEGL.so \
        "$@"
else
    echo ">> Reusing existing configure in $BUILD (pass -D... args after --)"
    cmake -S "$SRC" -B "$BUILD" "$@"
fi

echo ">> Building target: $TARGET ($JOBS jobs)"
cmake --build "$BUILD" --target "$TARGET" -j"$JOBS"

EYEPIECE="$BUILD/tools/cpp-tools/map-rasterizer/eyepiece"
echo ">> EyePiece ready: $EYEPIECE"
