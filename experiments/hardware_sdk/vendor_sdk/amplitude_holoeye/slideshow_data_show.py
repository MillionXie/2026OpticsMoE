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


# Plays a slideshow on the SLM with live-calculated 1d or 2d float (32-bit float)
# phase value data fields consisting of vertical blazed gratings with different periods.
# Each frame is shown on the SLM as soon as the data was transmitted,
# using the API function slm.showPhasevalues().
# The visible duration of each phase value field can be configured, and the slideshow
# will run asynchronously with the video interface frame rate, i.e. the show function
# will not block until the data is visible, and the visible state will be reached at
# next vertical synchronization (vsync) on the video signal. If a visible duration is configured,
# which does not fit into device vsync frame times, the visible duration of the phase
# fields will adapt accordingly, so that the selected visible duration is fulfilled on average.

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
# This might have an impact on performance, especially in "Capture SLM screen" mode.
# Please adapt the file showSLMPreview.py if preview window is not at the right position or even not visible.
if True:  # Please disable (False) to reach best timings.
    from showSLMPreview import showSLMPreview
    showSLMPreview(slm, scale=1.0)

# Configure slideshow (steer the laser beam from left to right and back):
gratingPeriodMin = 8
gratingPeriodMax = 64
gratingPeriodStepSize = 4
dataDisplayDurationInMilliSec = 100.0  # duration of each data frame in ms
repeatSlideshow = 3  # <= 0 (e. g. -1) repeats until Python process gets killed
printDetailedTimings = True
dataWidth = slm.width_px
dataHeight = 1  # slm.height_px

print("dataDisplayDurationInMilliSec = " + str(dataDisplayDurationInMilliSec))
print("repeatSlideshow = " + str(repeatSlideshow))
print("dataWidth = " + str(dataWidth))
print("dataHeight = " + str(dataHeight))

gratingPeriodList = []
gratingPeriodList += range(-gratingPeriodMin, -(gratingPeriodMax+1), -gratingPeriodStepSize)
gratingPeriodList += range(gratingPeriodMax, gratingPeriodMin-1, -gratingPeriodStepSize)
gratingPeriodList += range(gratingPeriodMin, gratingPeriodMax+1, gratingPeriodStepSize)
gratingPeriodList += range(-gratingPeriodMax, -(gratingPeriodMin-1), gratingPeriodStepSize)

print("gratingPeriodList = " + str(gratingPeriodList))
print("len(gratingPeriodList) = " + str(len(gratingPeriodList)))

# Calculate video signal frame time:
videoSignalFrameDurationMs = 1000.0 / slm.refreshrate_hz
# Check how many video signal frames were requested:
dataDisplayDurationInFrames = dataDisplayDurationInMilliSec / videoSignalFrameDurationMs
print("dataDisplayDurationInFrames = " + str("%7.3f.\n" % dataDisplayDurationInFrames))
# Check whether actual display duration fits into optimal hardwareFrameDurationMs at least within 0.1 ms:
if abs(dataDisplayDurationInMilliSec - round(dataDisplayDurationInFrames)*videoSignalFrameDurationMs) > 0.1:
    print("\nWARNING: This example runs asychron to the actual video interface frames.\n" +
          "           Therefore, reported timings may be precise, but on video signal\n" +
          "           output display durations will vary over time, without screen tearing.\n" +
          "           To avoid this, please use advanced API instead and set the\n" +
          "           durationInFrames property of the data handles, see example\n" +
          "           slideshow_data_preload.py.\n\n")

# Prepare phase value field:
blazeData = slmdisplaysdk.createFieldSingle(dataWidth, dataHeight)
phaseModulation = 2.0*math.pi  # radian

# Play slideshow:
print("Playing images on SLM. Press Ctrl-C to exit from playback.")

# Save the start time:
avgFPSTotalStartTime = slm.timeNow()
lastVisibleStartTime = avgFPSTotalStartTime

# Prepare playback measurements and output:
n = 0
frameCountTotal = 0
nextPrintStr = None
# Repeat slideshow playback:
while (n < repeatSlideshow) or (repeatSlideshow <= 0):
    n += 1

    avgFPSPerRunStartTime = slm.timeNow()
    frameCount = 0

    # Play slideshow once:
    for blazePeriod in gratingPeriodList:
        frameCount += 1
        frameCountTotal += 1

        calcStartTime = slm.timeNow()

        # Calculate data:
        for y in range(dataHeight):
            row = blazeData[y]
            for x in range(dataWidth):
                # Create a linear phase ramp with zero phase shift in the center.
                # Add a half gray value (0.5/256) to avoid rounding issues at the wrapping boundary.
                row[x] = float(phaseModulation * (x - dataWidth/2.0) / blazePeriod + phaseModulation * 0.5/256.0)

        calcEndTime = slm.timeNow()
        calcDurationMs = slm.timeDurationMs(calcEndTime, calcStartTime)

        # Wait to reach desired visible duration:
        currentVisibleDurationMs = slm.timeDurationMs(calcEndTime, lastVisibleStartTime)
        waitDurationMs = dataDisplayDurationInMilliSec - currentVisibleDurationMs
        if frameCountTotal > 1 and waitDurationMs > 0.0:
            waitDurationMeasuredMs = slm.timeWaitMs(waitDurationMs)

            if nextPrintStr is not None:
                nextPrintStr += ", wait("+str("%6.1f" % waitDurationMs)+" ms) took " + str("%6.1f" % waitDurationMs) + " ms"

        # Measure the time point before the call to slm.showPhasevalues(), so that we know the visible time of the last shown data:
        visibleStartTime = slm.timeNow()
        lastVisibleDurationMs = slm.timeDurationMs(visibleStartTime, lastVisibleStartTime)
        lastVisibleStartTime = visibleStartTime

        if nextPrintStr is not None:
            nextPrintStr += ", visible duration = " + str("%6.1f" % lastVisibleDurationMs) + " ms"
        if frameCountTotal > 1 and nextPrintStr is not None and printDetailedTimings:
            print(nextPrintStr)

        # Show data on SLM as soon as possible:
        error = slm.showPhasevalues(blazeData)
        assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

        nextPrintStr = "n = " + str("%3d" % n) + ", blazePeriod = " + str("%3d" % blazePeriod) + " px"
        nextPrintStr += ", calc. took " + str("%5.1f" % calcDurationMs) + " ms"

    # Show average frames per second for last run:
    runTimeMs = slm.timeDurationMs(slm.timeNow(), avgFPSPerRunStartTime)
    # Last shown frame has not finished its visible duration yet if this is the first most run of the slideshow:
    frameCountForRunTime = frameCount
    if n == 1:
        frameCountForRunTime = frameCount - 1
    print('--- Average FPS (n = ' + str("%3d" % n) + ') = ' +
          str("%6.2f" % (float(frameCountForRunTime) * 1000.0 / runTimeMs)) +
          str(' / Average frame time = %7.2f ms. ---' % (runTimeMs / float(frameCountForRunTime)))
          )

# Show average frames per second for all runs:
totalRunTimeMs = slm.timeDurationMs(slm.timeNow(), avgFPSTotalStartTime)
print('--- Average FPS (total) =   %6.2f ---' % (float(frameCountTotal-1) * 1000.0 / totalRunTimeMs) +
      str(' / Average frame time = %7.2f ms. ---' % (totalRunTimeMs / float(frameCountTotal-1)))
      )

# Show a blank screen with gray value 128 as last image:
error = slm.showBlankscreen(128)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Wait until the SLM process is closed:
print("Waiting for SDK process to close. Please close the tray icon to continue ...")
error = slm.utilsWaitUntilClosed()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# Unloading the SDK may or may not be required depending on your IDE:
slm = None
