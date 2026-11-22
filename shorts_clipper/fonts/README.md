# Caption display fonts

These fonts come from [google/fonts](https://github.com/google/fonts) and are
redistributed under the SIL Open Font License 1.1. Each family's license is
included beside its TTF, with the original copyright attribution.

- **Montserrat Black**: [Montserrat[wght].ttf](https://github.com/google/fonts/blob/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf),
  instantiated at weight 900 to a static `Montserrat-Black.ttf` so Pillow and
  libass use the same heavy face without depending on variable-font support.
  License: `Montserrat-OFL.txt`.
- **Anton**: [Anton-Regular.ttf](https://github.com/google/fonts/blob/main/ofl/anton/Anton-Regular.ttf).
  License: `Anton-OFL.txt`.
- **Bebas Neue**: [BebasNeue-Regular.ttf](https://github.com/google/fonts/blob/main/ofl/bebasneue/BebasNeue-Regular.ttf).
  License: `BebasNeue-OFL.txt`.

Montserrat was generated with FontTools 4.68.0 (a build-time tool only):

```sh
fonttools varLib.instancer 'Montserrat-Variable.ttf' wght=900 --update-name-table --output Montserrat-Black.ttf
```

The public `Montserrat` option selects this Black face. The original system
fonts remain available for existing stored styles. Rendering supplies this
directory to libass via `fontsdir`; wrapping loads these exact TTF files.
