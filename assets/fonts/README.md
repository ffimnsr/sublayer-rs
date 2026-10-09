# Bundled Fonts

All fonts are licensed under the SIL Open Font License 1.1 (`OFL.txt` is
available at each upstream project). They are passed to libass explicitly via
`fontsdir` so captions render identically on every machine, even without
system font packs.

| File | Family | Upstream |
| --- | --- | --- |
| `Montserrat-ExtraBold.ttf` | Montserrat ExtraBold | <https://github.com/JulietaUla/Montserrat> |
| `Montserrat-Bold.ttf` | Montserrat Bold | <https://github.com/JulietaUla/Montserrat> |
| `Inter.ttf` | Inter (variable) | <https://github.com/google/fonts/tree/main/ofl/inter> |
| `SpaceGrotesk.ttf` | Space Grotesk (variable) | <https://github.com/google/fonts/tree/main/ofl/spacegrotesk> |
| `Karla.ttf` | Karla (variable) | <https://github.com/google/fonts/tree/main/ofl/karla> |

Refetching: `scripts/fetch-fonts.sh` re-downloads every file from its
upstream URL; run it to refresh or restore the directory.

Themes reference family names, not file names: `Montserrat`, `Inter`, and
`Space Grotesk`. libass resolves a family to the matching file in this
directory when one is present, and falls back to system fonts otherwise.