# Teacher-style 2x2 Fresnel phase array

This directory adapts the supplied MATLAB formula to the current phase SLM:

- `1920x1200` pixels;
- `8 um` pixel pitch;
- `532 nm` wavelength;
- optical center `(980,590)` in continuous pixel-edge coordinates;
- `2x2` identical `350x350` Fresnel tiles;
- `30 cm` focal length.

The exported BMP is 8-bit grayscale. Black outside the `700x700` footprint means
zero phase; it is not an amplitude aperture and does not block light. No vertical
or horizontal flip is applied because this symmetric 2x2 pattern is invariant to
the configured phase-export flip.

Run either implementation from the repository root:

```powershell
python experiments\hardware_sdk\artifacts\calibration\fresnel_30cm\generate_fresnel_2x2_30cm.py
```

or run `generate_fresnel_2x2_30cm.m` in MATLAB. Both implement the same centered
pixel-coordinate formula and output filename.

## 10 cm version

The otherwise identical 10 cm mask is generated with:

```powershell
python experiments\hardware_sdk\artifacts\calibration\fresnel_30cm\generate_fresnel_2x2_10cm.py
```

or by running `generate_fresnel_2x2_10cm.m` in MATLAB. It writes
`fresnel_2x2_f100mm_w350_center980_590_1920x1200.bmp` and keeps the same SLM
center, tile size, array footprint, wavelength, and pixel pitch.
