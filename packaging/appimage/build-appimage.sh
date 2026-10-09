#!/bin/sh
# Assembles an AppImage for the Sublayer studio and CLI.
#
# Requirements:
#   * a release build of the workspace (`cargo build --release`)
#   * `appimagetool` on PATH, or APPIMAGETOOL pointing at it
#   * optionally `linuxdeploy` (LINUXDEPLOY) to bundle host libraries
#
# Usage:
#   packaging/appimage/build-appimage.sh [output.AppImage]
#
# The script only ever writes to packaging/appimage/AppDir and the output
# file; nothing is installed system-wide. Network is only used when
# `linuxdeploy` itself fetches its plugins (skip it by leaving it unset).
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APPDIR="${APPDIR:-$SCRIPT_DIR/AppDir}"
OUTPUT="${1:-$ROOT/target/Sublayer-x86_64.AppImage}"
APPIMAGETOOL="${APPIMAGETOOL:-appimagetool}"
LINUXDEPLOY="${LINUXDEPLOY:-}"

cd "$ROOT"

if [ ! -x target/release/sublayer-ui ] || [ ! -x target/release/sublayer ]; then
    echo "building release binaries…"
    cargo build --release -p sublayer-ui -p sublayer-cli
fi

echo "assembling $APPDIR"
rm -rf "$APPDIR"
install -d \
    "$APPDIR/usr/bin" \
    "$APPDIR/usr/share/applications" \
    "$APPDIR/usr/share/icons/hicolor/scalable/apps" \
    "$APPDIR/usr/share/sublayer/fonts"

install -m755 target/release/sublayer-ui "$APPDIR/usr/bin/sublayer-ui"
install -m755 target/release/sublayer "$APPDIR/usr/bin/sublayer"
install -m644 packaging/sublayer.desktop "$APPDIR/usr/share/applications/com.vastorigins.Sublayer.desktop"
install -m644 packaging/sublayer.desktop "$APPDIR/com.vastorigins.Sublayer.desktop"
install -m644 assets/icons/hicolor/scalable/apps/com.vastorigins.Sublayer.svg \
    "$APPDIR/usr/share/icons/hicolor/scalable/apps/com.vastorigins.Sublayer.svg"
cp assets/icons/hicolor/scalable/apps/com.vastorigins.Sublayer.svg "$APPDIR/com.vastorigins.Sublayer.svg"
cp assets/fonts/*.ttf "$APPDIR/usr/share/sublayer/fonts/"
install -m755 packaging/appimage/AppRun "$APPDIR/AppRun"

if [ -n "$LINUXDEPLOY" ] && [ -x "$LINUXDEPLOY" ]; then
    echo "bundling host libraries with linuxdeploy"
    "$LINUXDEPLOY" --appdir "$APPDIR" \
        --desktop-file "$APPDIR/com.vastorigins.Sublayer.desktop" \
        --icon-file "$APPDIR/com.vastorigins.Sublayer.svg"
fi

if ! command -v "$APPIMAGETOOL" >/dev/null 2>&1; then
    echo "appimagetool not found; AppDir is ready at $APPDIR" >&2
    echo "install appimagetool or set APPIMAGETOOL, then run:" >&2
    echo "  ARCH=x86_64 $APPIMAGETOOL $APPDIR $OUTPUT" >&2
    exit 1
fi

echo "packing $OUTPUT"
ARCH="${ARCH:-x86_64}" "$APPIMAGETOOL" "$APPDIR" "$OUTPUT"
echo "wrote $OUTPUT"
