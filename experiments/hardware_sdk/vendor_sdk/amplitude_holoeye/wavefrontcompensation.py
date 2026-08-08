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


# Loads a demo wavefront compensation file and shows it on the SLM.

# Import the SLM Display SDK:
import os
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
# The additional flag ShowWavefrontCompensation presses
# the button to show the wavefront compensation in preview window from code.
from showSLMPreview import showSLMPreview
showSLMPreview(slm, scale=0.0, flags=slmdisplaysdk.SLMPreviewFlags.ShowWavefrontCompensation)

# Configure the blank screen:
grayValue = 128

# Set the used incident laser wavelength in nanometer:
laser_wavelength_nm = 532.0

# Show gray value on SLM:
error = slm.showBlankscreen(grayValue)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

thisScriptPath = os.path.dirname(__file__)
wavefrontfile = os.path.join(thisScriptPath, "data", "wfcdemo_holoeye_logo.h5")
error = slm.wavefrontcompensationLoad(wavefrontfile, laser_wavelength_nm, slmdisplaysdk.WavefrontcompensationFlags.NoFlag, 0, 0)
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
