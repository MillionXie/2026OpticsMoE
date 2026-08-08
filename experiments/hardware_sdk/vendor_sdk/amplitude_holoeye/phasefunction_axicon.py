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


# Calculates an axicon using numpy and show it on the SLM.

import sys, math

# Import the SLM Display SDK:
import detect_heds_module_path
from holoeye import slmdisplaysdk

if not slmdisplaysdk.supportNumPy:
    print("Please install numpy to make this example work on your system.")
    sys.exit()

# Import numpy for matrix multiplication:
import numpy as np

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
dataWidth =  slm.width_px
dataHeight = slm.height_px

x = np.linspace(1, dataWidth, dataWidth, dtype=np.float32) - np.float32(dataWidth/2) - np.float32(centerX)
y = np.linspace(1, dataHeight, dataHeight, dtype=np.float32) - np.float32(dataHeight/2) - np.float32(centerY)

x2 = np.matrix(x*x)
y2 = np.matrix(y*y).transpose()

phaseData = np.float32(phaseModulation) * np.sqrt(np.array((np.dot(np.ones([dataHeight, 1], np.float32), x2) + np.dot(y2, np.ones([1, dataWidth], np.float32))), dtype=np.float32), dtype=np.float32) / np.float32(innerRadius)

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
