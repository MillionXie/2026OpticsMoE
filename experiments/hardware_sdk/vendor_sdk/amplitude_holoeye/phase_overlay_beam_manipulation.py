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
# Then we use the beam manipulation provided through the data handles to apply a phase overlay (tip/tilt/lens/offset).

import math

# Import the SLM Display SDK:
import detect_heds_module_path
from holoeye import slmdisplaysdk

# Initializes the SLM library:
slm = slmdisplaysdk.SLMInstance()

# Check if the library implements the required version:
if not slm.requiresVersion(5):
    exit(1)

# Detect SLMs and open a window on the selected SLM:
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Open the SLM preview window in non-scaled mode:
# Please adapt the file showSLMPreview.py if preview window
# is not at the right position or even not visible.
from showSLMPreview import showSLMPreview
showSLMPreview(slm, scale=1.0)

# Configure the blank screen where overlay is applied to:
grayValue = 128
grayValueOffset = -128  # this value is applied also as a beam manipulation later.

# Configure beam manipulation in physical units:
wavelength_nm = 633.0  # wavelength of incident laser light

steering_angle_x_deg = 0.2
steering_angle_y_deg = -0.3
focal_length_mm = 200.0

# Upload a datafield into the GPU. The datafield just consists of a single pixel with the grayValue and will
# automatically be extended into full SLM screen due to "PresetAutomatic" show flag.
# The loadData() call creates a handle which refers to the grayValue data.
grayData = slmdisplaysdk.createFieldUChar(1, 1)  # we need to have a numpy or ctypes array to pass data to loadData() function.
grayData[0][0] = grayValue

error, handle = slm.loadData(grayData)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Wait for the data to be processed internally to speed up the slm.showDatahandle(handle) call later:
error = slm.datahandleWaitFor(handle, slmdisplaysdk.State.ReadyToRender)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Make the data without overlay visible on SLM screen:
error = slm.showDatahandle(handle)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

print("Unmodified data is now visible on SLM.")

# Wait 2 seconds until we apply the beam manipulation to make the uploaded data visible first:
slm.utilsWaitForS(2.0)

beam_lens_param = slm.utilsBeamLensFromFocalLengthMM(wavelength_nm, focal_length_mm)
beam_steer_x_param = slm.utilsBeamSteerFromAngleDeg(wavelength_nm, steering_angle_x_deg)
beam_steer_y_param = slm.utilsBeamSteerFromAngleDeg(wavelength_nm, steering_angle_y_deg)

print("beam_steer_x_param = "+str(beam_steer_x_param)+" ==> steering angle x = " + str(slm.utilsBeamSteerToAngleDeg(wavelength_nm, beam_steer_x_param)) + " deg")
print("beam_steer_y_param = "+str(beam_steer_y_param)+" ==> steering angle y = " + str(slm.utilsBeamSteerToAngleDeg(wavelength_nm, beam_steer_y_param)) + " deg")
print("beam_lens_param = "+str(beam_lens_param)+" ==> f = " + str(slm.utilsBeamLensToFocalLengthMM(wavelength_nm, beam_lens_param)) + " mm")


# Now, after loadData(), we apply values to the beam manipulation parameters of the handle:
handle.beamSteerX = beam_steer_x_param
handle.beamSteerY = beam_steer_y_param
handle.beamLens = beam_lens_param

# The handle.valueOffset is given in float gray values (like in slm.showData() when passing float values).
# Actually addressed 8-bit gray values in range [0, 255] translate to float gray values in range [0.0, 1.0].
# Both ranges typically translate to a phase shift in range [0 rad, (255/256)*2pi rad] due to the phase calibration
# of the SLM and to make the addressed phase values periodic, i.e. gray value 256 must equal gray value 0.
handle.valueOffset = float(grayValueOffset)/255.0

# Apply the beam steering values from the handle structure to the SLM Display SDK.
# This will take effect on SLM screen directly, because we made the handle visible before applying values.
# Of course we also can apply the parameters before showing the handle on screen.
# We explicitly pass which values to apply by using the "ApplyDataHandleValue" flags:
error = slm.datahandleApplyValues(handle, slmdisplaysdk.ApplyDataHandleValue.BeamManipulation | slmdisplaysdk.ApplyDataHandleValue.ValueOffset)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

print("Data is now overlayed with phase functions generated from given beam manipulation values.")

# Now the data should have changed by our beam manipulations.


# If your IDE terminates the python interpreter process after the script is finished, the SLM content
# will be lost as soon as the script finishes.

# You may insert further code here.

# Wait until the SLM process is closed:
print("Waiting for SDK process to close. Please close the tray icon to continue ...")
error = slm.utilsWaitUntilClosed()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Unloading the SDK may or may not be required depending on your IDE:
slm = None
