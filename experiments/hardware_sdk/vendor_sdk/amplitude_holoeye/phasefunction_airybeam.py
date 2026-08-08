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


# Calculates an airy beam using numpy and show it on the SLM.

import sys, math

# Import the SLM Display SDK:
import detect_heds_module_path
from holoeye import slmdisplaysdk

if slmdisplaysdk.supportNumPy:
    # Import numpy for matrix multiplication:
    import numpy as np


def computeAiry(phaseModulation, dataWidth, dataHeight, centerX , centerY, innerRadius, rotAngleDeg):
    angleRad = rotAngleDeg / 360.0 * 2.0 * math.pi

    # Reserve memory for the phase data matrix.
    # Use data type single to optimize performance:
    phaseData = slmdisplaysdk.createFieldSingle(dataWidth, dataHeight, False)

    for y in range(dataHeight):
        row = phaseData[y]

        for x in range(dataWidth):
            x0Deg = x - dataWidth / 2 - centerX
            y0Deg = y - dataHeight / 2 + centerY

            xDeg = math.cos(angleRad) * x0Deg - math.sin(angleRad) * y0Deg
            yDeg = math.sin(angleRad) * x0Deg + math.cos(angleRad) * y0Deg

            x3 = math.pow(xDeg, 3.0)
            y3 = math.pow(yDeg, 3.0)

            if onedimensional:
                val = x3 * math.pow(innerRadius, -3.0)
            else:
                val = (x3 + y3) * math.pow(innerRadius, -3.0)

            row[x] = val * phaseModulation + cyclicShift

    return phaseData


def computeAiryNumPy (phaseModulation, dataWidth, dataHeight, centerX, centerY, innerRadius):
    x = np.linspace(1, dataWidth, dataWidth, dtype=np.float32) - np.float32(dataWidth / 2) - np.float32(centerX)
    y = np.linspace(1, dataHeight, dataHeight, dtype=np.float32) - np.float32(dataHeight / 2) + np.float32(centerY)
    x3 = np.matrix(np.power(x, 3, dtype=np.float32))
    y3 = np.matrix(np.power(y, 3, dtype=np.float32)).transpose()

    if onedimensional:
        ar = np.array((np.dot(np.ones([dataHeight, 1], np.float32), x3)), dtype=np.float32)
    else:
        ar = np.array((np.dot(np.ones([dataHeight, 1], np.float32), x3) + np.dot(y3, np.ones([1, dataWidth], np.float32))), dtype=np.float32)

    phaseData = ar * np.float32(phaseModulation) * np.power(innerRadius, -3, dtype=np.float32) + cyclicShift

    return phaseData


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
from showSLMPreview import showSLMPreview
showSLMPreview(slm, scale=0.0)

# Configure the airy beam properties:
onedimensional = False
innerRadius = slm.height_px / 2.0
rotAngleDeg = 0.0
centerX = 0
centerY = 0

# Calculate the phase values of an airy beam in a pixel-wise matrix:

# pre-calc. helper variables:
phaseModulation = 2*math.pi
dataWidth = slm.width_px
dataHeight = slm.height_px

# Move white-black phase wraps out of the center:
cyclicShift = - phaseModulation / 2.0

if rotAngleDeg != 0.0 or not slmdisplaysdk.supportNumPy:
    print("using slower for-loop")
    phaseData = computeAiry(phaseModulation, dataWidth, dataHeight, centerX , centerY, innerRadius, rotAngleDeg)
else:
    print("using numpy")
    phaseData = computeAiryNumPy(phaseModulation, dataWidth, dataHeight, centerX, centerY, innerRadius)

# Show data on the SLM:
error = slm.showPhasevalues(phaseData)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# If your IDE terminates the python interpreter process after the script is finished, the SLM content
# will be lost as soon as the script finishes.

# You may insert further code here.

# Wait until the SLM process is closed:
print("Waiting for SDK process to close. Please close the tray icon to continue ...")
error = slm.utilsWaitUntilClosed()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Unloading the SDK may or may not be required depending on your IDE:
slm = None
