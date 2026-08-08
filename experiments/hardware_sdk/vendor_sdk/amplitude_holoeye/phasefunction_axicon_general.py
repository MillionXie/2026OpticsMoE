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


# Calculates an axicon and shows it on the SLM.

import math

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
from showSLMPreview import showSLMPreview
showSLMPreview(slm, scale=0.0)

# Configure the axicon properties:
innerRadius = slm.height_px / 3
centerX = 0
centerY = 0

# Calculate the phase values of an axicon in a pixel-wise matrix:

# pre-calc. helper variables:
phaseModulation = 2*math.pi
dataWidth = slm.width_px
dataHeight = slm.height_px

# Reserve memory for the phase data matrix.
# Use data type single to optimize performance:
phaseData = slmdisplaysdk.createFieldSingle(dataWidth, dataHeight)

for y in range(dataHeight):
    row = phaseData[y]
    y2 = math.pow(y - dataHeight/2 + centerY, 2)

    for x in range(dataWidth):
        x2 = math.pow(x - dataWidth/2 - centerX, 2)

        r = math.sqrt(x2 + y2)

        row[x] = phaseModulation * r / innerRadius

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
