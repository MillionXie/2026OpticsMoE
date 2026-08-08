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


# Loads an RGB image file into a data handle, applies some transformations
# (shift x, y, and scale in x and y), and shows it on the SLM.

import os

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
# Load image file to GPU:
thisScriptPath = os.path.dirname(__file__)
filename = os.path.join(thisScriptPath, "data", "amp_white_400.bmp")

error, handle = slm.loadDataFromFile(filename)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Wait for the data to be processed internally to speed up the slm.showDatahandle(handle) call later:
error = slm.datahandleWaitFor(handle, slmdisplaysdk.State.ReadyToRender)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Make the data with overlay visible on SLM screen without any transform:
error = slm.showDatahandle(handle, slmdisplaysdk.ShowFlags.PresentAutomatic)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

print("Unmodified data is now visible on SLM.")

# # Wait 2 seconds until we apply transformations to make the loaded data visible first:
# slm.utilsWaitForS(2.0)
#
# # Now, after loadData(), we apply values to the transform parameters of the handle:
# handle.transformShiftX = slm.width_px//4
# handle.transformShiftY = slm.height_px//4
# handle.transformScale = .5
#
# # Apply the transform values from the handle structure to the SLM Display SDK.
# # This will take effect on SLM screen directly, because we made the handle visible before applying values.
# # Of course we also can apply the parameters before showing the handle on screen.
# # We explicitly pass which values to apply by using the "ApplyDataHandleValue" flags:
# error = slm.datahandleApplyValues(handle, slmdisplaysdk.ApplyDataHandleValue.Transform)
# assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

print("Data is now manipulated by given transformations.")

# Now the data should have changed by our transformations.


# If your IDE terminates the python interpreter process after the script is finished, the SLM content
# will be lost as soon as the script finishes.

# You may insert further code here.

# Wait until the SLM process is closed:
print("Waiting for SDK process to close. Please close the tray icon to continue ...")
error = slm.utilsWaitUntilClosed()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Unloading the SDK may or may not be required depending on your IDE:
slm = None
