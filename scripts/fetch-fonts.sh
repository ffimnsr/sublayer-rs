#!/bin/sh
# Re-downloads the bundled caption fonts from their upstream sources.
# All files are SIL OFL 1.1 licensed, except KomikaAxis.ttf which is
# freeware and vendored manually. See assets/fonts/README.md.
set -eu

cd "$(dirname "$0")/../assets/fonts"

curl -fsSL -o Montserrat-ExtraBold.ttf \
    https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf
curl -fsSL -o Montserrat-Bold.ttf \
    https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf
curl -fsSL -o Inter.ttf \
    "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf"
curl -fsSL -o SpaceGrotesk.ttf \
    "https://raw.githubusercontent.com/google/fonts/main/ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf"
curl -fsSL -o Karla.ttf \
    "https://raw.githubusercontent.com/google/fonts/main/ofl/karla/Karla%5Bwght%5D.ttf"

echo "fonts refreshed"