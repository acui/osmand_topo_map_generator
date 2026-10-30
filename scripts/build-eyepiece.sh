#!/usr/bin/env bash
# Build the EyePiece rasterizer from UNMODIFIED official OsmAnd repos plus the
# patch overlay in patches/ (applied automatically, idempotently).
#
# Layout:
#   osmand-native/core, tools, OsmAnd-resources   official checkouts (submodules)
#   patches/<repo>/*.patch                        this project's modifications
#
# Usage:
#   scripts/build-eyepiece.sh
#   JOBS=8 BUILD=/path/to/build scripts/build-eyepiece.sh
#
# Requirements (Debian/Ubuntu): build-essential cmake ninja-build git
#   libgdal-dev libgl1-mesa-dev libegl1-mesa-dev qtbase5-dev
#   (everything else is bundled in core/externals)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # generator repo
SRC="${NATIVE:-$ROOT/osmand-native}"                       # native checkouts (submodules)
BUILD="${BUILD_DIR:-$SRC/build}"
JOBS="${JOBS:-$(nproc)}"
TARGET="${TARGET:-eyepiece}"

if [ ! -d "$SRC/core" ]; then
    echo "osmand-native submodule missing. Run: git submodule update --init" >&2
    exit 1
fi

# --- apply the patch overlay onto the official checkouts (idempotent) --------
apply_patches() {
    local repo="$1" name="$2"
    local patches="$ROOT/patches/$name"
    [ -d "$patches" ] || return 0
    cd "$repo"
    if git apply --check --reverse "$patches"/*.patch 2>/dev/null; then
        echo ">> [$name] patches already applied"
        return 0
    fi
    if git apply --check "$patches"/*.patch 2>/dev/null; then
        echo ">> [$name] applying patch overlay"
        git apply "$patches"/*.patch
    else
        echo ">> [$name] WARNING: patches neither applied nor applicable - skipping" >&2
    fi
}

apply_patches "$SRC/core" core
apply_patches "$SRC/tools" tools
apply_patches "$SRC/OsmAnd-resources" OsmAnd-resources

# --- prepare external dependency sources (fresh clones have none) ------------
for ext in "$SRC/core/externals"/*/; do
    if [ -x "$ext/configure.sh" ]; then
        echo ">> preparing external: $(basename "$ext")"
        (cd "$ext" && bash configure.sh)
    fi
done
cd "$SRC"

# --- configure + build --------------------------------------------------------
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    echo ">> Configuring ($BUILD)"
    # -O1 matches the known-good configuration; the font/style directories are
    # required so the embedded resource bundle carries them (without them
    # initializeRendering() fails at runtime).
    cmake -S "$SRC" -B "$BUILD" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG" \
        -DBUNDLE_FONTS_DIR="${FONTS_DIR:-$HOME/osmand-data/fonts}" \
        -DBUNDLE_STYLES_DIR="$ROOT/styles" \
        "$@"
else
    echo ">> Reusing existing configure in $BUILD"
    cmake -S "$SRC" -B "$BUILD" "$@"
fi

echo ">> Building target: $TARGET ($JOBS jobs)"
cmake --build "$BUILD" --target "$TARGET" -j"$JOBS"

# EyePiece loads the shared resource bundle at runtime (non-null provider)
echo ">> Building resource bundle"
cmake --build "$BUILD" --target OsmAndCore_ResourcesBundle_shared -j"$JOBS"

EYEPIECE="$BUILD/tools/cpp-tools/map-rasterizer/eyepiece"
echo ">> EyePiece ready: $EYEPIECE"
