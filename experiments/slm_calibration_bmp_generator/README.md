# SLM calibration BMP generator

This small experiment creates centered, 8-bit grayscale BMP patterns for the
current hardware. The active region is 956×956 pixels at 8 µm, centered on a
1920×1080 amplitude SLM or a 1920×1200 phase SLM. Phase patterns are vertically
flipped before export to compensate the current folded optical path; amplitude
patterns are kept in the original orientation.

Generated amplitude patterns include uniform fields, checkerboard, crosshair,
letter A and a circular aperture. Phase patterns include flat 0/π, 0/π
checkerboard, crosshair, letter A, and 5 cm / 10 cm thin-lens phases at 532 nm.
All geometry and focal lengths are editable in the YAML file.

