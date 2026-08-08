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


# Plays a slideshow on the SLM with pre-loaded image files from a single folder.
# The image files are pre-loaded to the GPU once, and then each image is shown on the SLM by selecting the
# appropriate ID of the pre-laoded file on the GPU to reach higher performance.
# The duration each image is shown can be configured and is maintained by the GPU as much as possible.
# For holograms, please use image formats which are uncompressed (e.g. BMP) or which use lossless compression, like PNG.

import os, sys, time

# Import the SLM Display SDK:
import detect_heds_module_path
from holoeye import slmdisplaysdk

# Import helper function to print timing statistics of the display duration of the handles:
from slideshow_preload_print_stats import printStat

# Initializes the SLM library
slm = slmdisplaysdk.SLMInstance()

# Check if the library implements the required version
if not slm.requiresVersion(5):
    exit(1)

# Detect SLMs and open a window on the selected SLM
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Open the SLM preview window in non-scaled mode:
# This might have an impact on performance, especially in "Capture SLM screen" mode.
# Please adapt the file showSLMPreview.py if preview window is not at the right position or even not visible.
from showSLMPreview import showSLMPreview
showSLMPreview(slm, scale=1.0)

# Configure slideshow:
thisScriptPath = os.path.dirname(__file__)
imageFolder = os.path.join(thisScriptPath, "data", "vertical_grating")
imageDisplayDurationMilliSec = 100  # please select the duration in ms each image file shall be shown on the SLM
repeatSlideshow = 3  # <= 0 (e. g. -1) repeats until Python process gets killed

# Please select how to scale and transform image files while displaying:
displayOptions = slmdisplaysdk.ShowFlags.PresentAutomatic  # PresentAutomatic == 0 (default)
#displayOptions |= slmdisplaysdk.ShowFlags.TransposeData
#displayOptions |= slmdisplaysdk.ShowFlags.PresentTiledCentered  # This makes much sense for holographic images
#displayOptions |= slmdisplaysdk.ShowFlags.PresentFitWithBars
#displayOptions |= slmdisplaysdk.ShowFlags.PresentFitNoBars
#displayOptions |= slmdisplaysdk.ShowFlags.PresentFitScreen

# Search image files in given folder:
filesList = os.listdir(imageFolder)

# Filter *.png, *.bmp, *.gif, and *.jpg files:
imagesList = [filename for filename in filesList if str(filename).endswith(".png") or str(filename).endswith(".gif") or str(filename).endswith(".bmp") or str(filename).endswith(".jpg")]

imagesList.sort()

print(imagesList)

print("Number of images found in imageFolder = " + str(len(imagesList)))

if len(imagesList) <= 0:
    sys.exit()


# Upload image data to GPU:
print("Loading data ...")
start_time = time.time()

durationInFrames = int((float(imageDisplayDurationMilliSec)/1000.0) * slm.refreshrate_hz)
if durationInFrames <= 0:
    durationInFrames = 1  # The minimum duration is one video frame of the SLM

print("slm.refreshrate_hz = " + str(slm.refreshrate_hz))
print("durationInFrames = " + str(durationInFrames))

dataHandles = []
calcPercent = -1

nHandle = 0  # total number of images loaded to GPU
for filename in imagesList:

    # Print progress:
    percent = int(float(nHandle) / len(imagesList) * 100)
    if int(percent / 5) > calcPercent:
        calcPercent = int(percent / 5)
        print(str(percent) + "%")

    filepath = os.path.join(imageFolder, filename)

    # Load image data to GPU:
    error, handle = slm.loadDataFromFile(filepath)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    error = slm.datahandleWaitFor(handle, slmdisplaysdk.State.LoadingFile)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    handle.durationInFrames = durationInFrames

    error = slm.datahandleApplyValues(handle, slmdisplaysdk.ApplyDataHandleValue.DurationInFrames)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    # Wait for actual upload of image data to GPU:
    slm.datahandleWaitFor(handle, slmdisplaysdk.State.ReadyToRender)

    nHandle += 1
    dataHandles.append(handle)

print("100%")
end_time = time.time()
print("Loading files took "+ str("%0.3f" % (end_time - start_time)) +" seconds\n")

# Play complete slideshow:
n = 0
while (n < repeatSlideshow) or (repeatSlideshow <= 0):
    n += 1

    print("Show images for the " + str(n) + ". time ...")

    # Play slideshow once:
    for handle in dataHandles:
        error = slm.showDatahandle(handle, displayOptions)
        assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    # Update the handles to the latest state:
    for handle in dataHandles:
        error = slm.updateDatahandle(handle)
        assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

    # Print the actual statistics (last data handle has wrong visible time before any other data was shown):
    dataHandlesNoLast = dataHandles[0:-1]
    print("Showing timing statistics...")
    printStat("delayTimeMs", dataHandlesNoLast)
    printStat("processingWaitTimeMs", dataHandlesNoLast)
    printStat("loadingTimeMs", dataHandlesNoLast)
    printStat("conversionTimeMs", dataHandlesNoLast)
    printStat("processingTimeMs", dataHandlesNoLast)
    printStat("transferTimeMs", dataHandlesNoLast)
    printStat("renderTimeMs", dataHandlesNoLast)
    printStat("becomeVisibleTimeMs", dataHandlesNoLast)
    printStat("visibleTimeMs", dataHandlesNoLast)


# One last image to clear the SLM screen after the slideshow playback:
# (Also possible by just calling slm.showBlankscreen(128))
data = slmdisplaysdk.createFieldUChar(1, 1)

if slmdisplaysdk.supportNumPy:
    data[0, 0] = 128
else:
    data[0][0] = 128

error, dh = slm.loadData(data)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

error = slm.showDatahandle(dh, slmdisplaysdk.ShowFlags.PresentAutomatic)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Release handles and their data to free up video memory:
dataHandles = None

# Wait until the SLM process is closed:
print("Waiting for SDK process to close. Please close the tray icon to continue ...")
error = slm.utilsWaitUntilClosed()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Unloading the SDK may or may not be required depending on your IDE:
slm = None
