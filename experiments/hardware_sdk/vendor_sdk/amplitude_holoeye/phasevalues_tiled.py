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


# Shows a 1d vector of phase values with data type float (single) on the SLM.
# The phase values have a range from 0 to 2pi, non-fitting values will be wrapped automatically on the GPU.
# We use the show-flags to replicate (tile) the 1d vector to the full SLM size.

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

# Open the SLM preview window in non-scaled mode:
# Please adapt the file showSLMPreview.py if preview window
# is not at the right position or even not visible.
from showSLMPreview import showSLMPreview
showSLMPreview(slm, scale=1.0)

# Calculate e.g. a horizontal blazed grating column:
blazePeriod = 77

phaseModulation = 2.0 * math.pi  # radian
dataWidth = blazePeriod
dataHeight = 1
phaseData = slmdisplaysdk.createFieldSingle(dataWidth, dataHeight)

# Calculate phase data. The values are calculated in unit radian without any wrapping:
dataRow = phaseData[0]
for x in range(dataWidth):
    # Create a linear phase ramp with zero phase shift in the center.
    # Add a half gray value (0.5/256) to avoid rounding issues at the wrapping boundary.
    dataRow[x] = float(phaseModulation * (x - dataWidth / 2.0) / blazePeriod + phaseModulation * 0.5/256.0)

# Show phase data on the SLM:
error = slm.showPhasevalues(phaseData, slmdisplaysdk.ShowFlags.PresentTiledCentered)
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
