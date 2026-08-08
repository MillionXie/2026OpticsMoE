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


# Loads an RGB image file using Python and shows the loaded data on the SLM.

import os

# Import the SLM Display SDK:
import detect_heds_module_path
from holoeye import slmdisplaysdk

if not slmdisplaysdk.supportNumPy:
    print("Please install NumPy and Python Image Library (PIL) into your Python distribution to make this example work on your system.")
    sys.exit()

from PIL import Image
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

# Show image file data on SLM:
thisScriptPath = os.path.dirname(__file__)
filename = os.path.join(thisScriptPath, "data", "RGBCMY01.png")

# Load image using PIL.Image:
image = Image.open(filename)

# Convert image into Numpy array:
imageData = np.asarray(image)

if True:
    # Show data directly:
    error = slm.showData(imageData, slmdisplaysdk.ShowFlags.PresentAutomatic)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
else:
    # Load data into GPU and show it with handle:
    error, handle = slm.loadData(imageData)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    error = slm.showDatahandle(handle, slmdisplaysdk.ShowFlags.PresentAutomatic)
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
