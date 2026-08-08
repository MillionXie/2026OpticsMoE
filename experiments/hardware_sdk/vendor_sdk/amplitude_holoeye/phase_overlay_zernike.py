# -*- coding: utf-8 -*-

#--------------------------------------------------------------------#
#                                                                    #
# Copyright (C) 2023 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
#                                                                    #
#--------------------------------------------------------------------#


# Uses the built-in blank screen function to show a given grayscale value on the full SLM.
# Then we use the Zernike functions as an overlay.

import ctypes

# Import the SLM Display SDK:
import detect_heds_module_path
from holoeye import slmdisplaysdk

# Initializes the SLM library
slm = slmdisplaysdk.SLMInstance()

# Check if the library implements the required version
if not slm.requiresVersion(5):
    exit(1)

# Detect SLMs and open a window on the selected SLM
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Open the SLM preview window in "Fit" mode:
# Please adapt the file showSLMPreview.py if preview window
# is not at the right position or even not visible.
# The additional flag ShowZernikeRadius presses the button to
# show the Zernike radius visualization in preview window from code.
from showSLMPreview import showSLMPreview
showSLMPreview(slm, scale=0.0, flags=slmdisplaysdk.SLMPreviewFlags.ShowZernikeRadius)

# Configure the blank screen:
grayValue = 128

# Show gray value on SLM without using a handle:
error = slm.showBlankscreen(grayValue)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

print("Unmodified data is now visible on SLM.")

# Wait 2 seconds until we apply the zernike overlay to make the uploaded data visible first:
slm.utilsWaitForS(2.0)

zernikeRadius = slm.height_px / 2.0 + 0.5  # default Zernike radius in HOLOEYE Pattern Generator

# zernikeDataVector consists of:
# index 0: Zernike radius in pixel, instead of piston.
# index 1: Tip (blazed grating with deviation in x-direction).
# index 2: Tilt (blazed grating with deviation in y-direction).
# index 3: Second order astigmatism.
# index 4: Defocus (r^2). Has the same effect like a lens.
# ...
# index 14: QuadrafoilY.
#
# The vector does not need to hold all elements, it just must have the size up to the last non-zero element, e.g.
#zernikeDataVector = [zernikeRadius, 0.2, 0.1, 0.0, 1.0, 0.0, 0.0, 0.25]

# We also can create the vector in more general way and set the Zernike coefficients by their names
# (see heds_types.py and documentation for a list of all names and their related polynomials):
zernikeDataVector = (ctypes.c_float * slmdisplaysdk.ZernikeValues.COUNT)()
zernikeDataVector[slmdisplaysdk.ZernikeValues.RadiusPx] = zernikeRadius  # default is half diagonal of SLM in pixel.
zernikeDataVector[slmdisplaysdk.ZernikeValues.TiltX] = 0.2
zernikeDataVector[slmdisplaysdk.ZernikeValues.TiltY] = 0.1
zernikeDataVector[slmdisplaysdk.ZernikeValues.Defocus] = 1.0
zernikeDataVector[slmdisplaysdk.ZernikeValues.ComaX] = 0.25


error = slm.zernike(zernikeDataVector)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

print("Data is now overlayed with phase functions generated from given Zernike coefficients:")
# Print out all used Zernike values:
for i in range(slmdisplaysdk.ZernikeValues.COUNT):
    if i == 0:
        print("    R   = {:7.2f}".format(zernikeDataVector[i]))
    else:
        print("    C{:02d} = {:7.2f}".format(i, zernikeDataVector[i]))


# Now the previously shown blank screen was overlayed with the phase function related to the given Zernike parameters.
# The Zernike overlay is globally applied, i.e. all display functions will be overlayed by the corresponding phase function.

# If your IDE terminates the python interpreter process after the script is finished, the SLM content
# will be lost as soon as the script finishes.

# You may insert further code here.

# Wait until the SLM process is closed:
print("Waiting for SDK process to close. Please close the tray icon to continue ...")
error = slm.utilsWaitUntilClosed()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Unloading the SDK may or may not be required depending on your IDE:
slm = None
