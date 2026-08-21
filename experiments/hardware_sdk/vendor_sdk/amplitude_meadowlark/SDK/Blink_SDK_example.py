# Example usage of Blink_C_wrapper.dll
# Meadowlark Optics Spatial Light Modulators
# August 1, 2024

import os
import numpy
from ctypes import *
from scipy import misc
from time import sleep

# Load the DLL
# Blink_C_wrapper.dll. If you move the example, take DLLs from the SDK folder
# to the new folder such that all dependent DLLs can be found
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

# Open the image generation library
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\ImageGen")
image_lib = CDLL("ImageGen")

# Basic parameters for calling Create_SDK
num_boards_found = c_uint(0)
constructed_okay = c_uint(-1)
board_number = 1
wait_For_Trigger = 0 #image writes to the SLM hold off until an external trigger is received by the hardware
flip_immediate = 0 #only supported on the 1024
OutputPulseImageFlip = 0 #enables the hardware to generate an output pulse when new images data is loaded to the SLM
timeout_ms = 5000
center_x = 512
center_y = 512
VortexCharge = 3
fork = 0
RGB = 0


# Call the Create_SDK constructor
slm_lib.Create_SDK(byref(num_boards_found), byref(constructed_okay))

# Constructed_okay = 1 means success. If constructed okay is 0, then check to see the error. It could be that no
# SLM is attached. This is acceptable, the software will allow the user to run in simulation mode. Or,
# it could mean that the driver handle is already open by another program (i.e. Blink or the Cal Kit) or that
# there is a problem with the device driver.
if constructed_okay.value == 0:
    print ("Blink SDK did not construct successfully")

# This is the number of boards found. If more than one board is found, the software will allow you to interact with
# each board individually through the board number. If no SLM is found, and you are running in simulation mode
# then the num boards found will still be 1.
if num_boards_found.value == 1:
    print ("Blink SDK was successfully constructed")
    print ("Found %s SLM controller(s)" % num_boards_found.value)
    height = slm_lib.Get_image_height(board_number)
    width = slm_lib.Get_image_width(board_number)
    depth = slm_lib.Get_image_depth(board_number) #Bits per pixel
    Bytes = depth//8
    center_x = width//2
    center_y = height//2
	
    slm_lib.SetWaitForTrigger (board_number, wait_For_Trigger)
    slm_lib.SetFlipImmediate (board_number, flip_immediate)
    slm_lib.SetOutputPulse (board_number, OutputPulseImageFlip)

    ''' *** You should replace *bit_linear.LUT with your custom LUT file *** '''
    # but for now open a generic LUT that linearly maps input graylevels to output voltages
    ''' ***Using *bit_linear.LUT does NOT give a linear phase response*** '''
    if width == 1920:
        load_lut_status = slm_lib.Load_LUT_file(board_number, b"C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1920x1152_linearVoltage.LUT")
    if width == 1024:
        load_lut_status = slm_lib.Load_LUT_file(board_number, b"C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1024x1024_linearVoltage.LUT")
    if load_lut_status == 0:
        print("Error loading LUT file. Check LUT file and path.")
    # Create three vectors to hold values for three SLM images with a blank image, example images, and fill the wavefront correction with a blank
    BlankImage = numpy.zeros([width*height*Bytes], numpy.uint8, 'C')
    ImageOne = numpy.zeros([width*height*Bytes], numpy.uint8, 'C')
    ImageTwo = numpy.zeros([width*height*Bytes], numpy.uint8, 'C')
    WFC = numpy.zeros([width*height*Bytes], numpy.uint8, 'C')

    # Write a blank pattern to the SLM to get going
    retVal = slm_lib.Write_image(board_number, BlankImage.ctypes.data_as(POINTER(c_ubyte)), timeout_ms)
    if(retVal != 1):
            print ("DMA Failed")
    else:
        retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)
        if(retVal != 1):
            print ("ImageWriteComplete failed, trigger never received?")
        else:
            # Generate phase gradients
            VortexCharge = 5
            image_lib.Generate_LG(ImageOne.ctypes.data_as(POINTER(c_ubyte)), WFC.ctypes.data_as(POINTER(c_ubyte)), width, height, depth, VortexCharge, center_x, center_y, fork, RGB)
            VortexCharge = 3
            image_lib.Generate_LG(ImageTwo.ctypes.data_as(POINTER(c_ubyte)), WFC.ctypes.data_as(POINTER(c_ubyte)), width, height, depth, VortexCharge, center_x, center_y, fork, RGB)

            # Loop between our phase gradient images
            for i in range(0, 10):

                # Write image returns on DMA complete, ImageWriteComplete returns when the hardware
                # image buffer is ready to receive the next image. Breaking this into two functions is
                # useful for external triggers. It is safe to apply a trigger when Write_image is complete
                # and it is safe to write a new image when ImageWriteComplete returns
                retVal = slm_lib.Write_image(board_number, ImageOne.ctypes.data_as(POINTER(c_ubyte)), timeout_ms)
                if(retVal != 1):
                    print ("DMA Failed")
                    break
                else:
                # Check the buffer is ready to receive the next image
                    retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)
                    if(retVal != 1):
                        print ("ImageWriteComplete failed, trigger never received?")
                        break

                    sleep(1.0) # This is in seconds. IF USING EXTERNAL TRIGGERS, SET THIS TO 0

                    retVal = slm_lib.Write_image(board_number, ImageTwo.ctypes.data_as(POINTER(c_ubyte)), timeout_ms)
                    if(retVal != 1):
                        print ("DMA Failed")
                        break
                    else:
                        # Check the buffer is ready to receive the next image
                        retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)
                        if(retVal != 1):
                            print ("ImageWriteComplete failed, trigger never received?")
                            break
                    sleep(1.0) # This is in seconds. IF USING EXTERNAL TRIGGERS, SET THIS TO 0
    # Write a blank pattern to SLM after sequence has finished
    retVal = slm_lib.Write_image(board_number, BlankImage.ctypes.data_as(POINTER(c_ubyte)), timeout_ms)
    if(retVal != 1):
            print ("DMA Failed")
    # Always call Delete_SDK before exiting
    slm_lib.Delete_SDK()

