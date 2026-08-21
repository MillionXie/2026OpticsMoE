import os
import sys
import numpy
from ctypes import *
import csv
from scipy import misc
from time import sleep

# Load the DLL
# Blink_C_wrapper.dll, Blink_SDK.dll, ImageGen.dll, FreeImage.dll and wdapi1021.dll
# should all be located in the same directory as the program referencing the
# library
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink Plus\\SDK\\ImageGen")
image_lib = CDLL("ImageGen")

# Basic parameters for calling Create_SDK
num_boards_found = c_uint(0)
constructed_okay = c_uint(-1)
board_number = 1
ExternalTrigger = 0
FlipImmediate = 0
OutputPulseImageFlip = 0 # Enables the hardware to generate an output pulse when new images data is loaded to the SLM
timeout_ms = 1000


# Call the Create_SDK constructor
# Returns a handle that's passed to subsequent SDK calls
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
    print("Blink SDK was successfully constructed")
    print(f"Found {num_boards_found.value} SLM controller(s)")
    NumDataPoints = 256 # 8 bit controller
    NumRegions = 1 # For a global LUT NumRegions = 1, for a regional LUT NumRegions = 64
    RGB = 0 # PCIe interface never uses RGB images
    height = slm_lib.Get_image_height(board_number)
    width = slm_lib.Get_image_width(board_number)
    depth = slm_lib.Get_image_depth(board_number) # Bits per pixel
    Bytes = depth//8
    center_x = width//2
    center_y = height//2
	
    slm_lib.SetWaitForTrigger (board_number, ExternalTrigger)
    slm_lib.SetFlipImmediate (board_number, FlipImmediate)
    slm_lib.SetOutputPulse (board_number, OutputPulseImageFlip)

    # When you are calibrating you want to load a linear LUT. If you are checking a calibration, load your calibrated LUT
    if width == 1920:
        load_lut_status = slm_lib.Load_LUT_file(board_number, b"C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1920x1152_linearVoltage.LUT")
    if width == 1024:
        load_lut_status = slm_lib.Load_LUT_file(board_number, b"C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1024x1024_linearVoltage.LUT")
    if load_lut_status == 0:
        print("Error loading LUT file. Check LUT file and path.")
    else:
        # When calibrating a LUT **ALWAYS** leave the WFC blank
        WFC = numpy.zeros([width*height*Bytes], numpy.uint8, 'C')

        # Create two arrays to hold the image data
        Blank = numpy.zeros([width*height*Bytes], numpy.uint8, 'C')
        Image = numpy.zeros([width*height*Bytes], numpy.uint8, 'C')

        # Create an array to hold measurements from the analog input (AI) board
        AI_Intensities = numpy.zeros([NumDataPoints], numpy.uint8, 'C')

        # Start the SLM with a blank image
        retVal = slm_lib.Write_image(board_number, Blank.ctypes.data_as(POINTER(c_ubyte)), timeout_ms)
        if(retVal != 1):
            print ("DMA Failed")
        else:
            # Check the buffer is ready to receive the next image
            retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)
            if(retVal != 1):
                print ("ImageWriteComplete failed, trigger never received?")

            # Load diffraction patterns, and record the 1st order intensity
            ''' ************* SET YOUR VARIABLES BELOW ACCORDINGLY **************** '''
            # 1920x1152 8 in/12 out: reference should be 0, increment variable from 0 to 255 with StepBy = +1
            # 1024x1024 8 in/12 out: reference should be 255, decrement variable from 255 down to 0 with StepBy = -1
            Reference = 255
            Variable = 255
            StepBy = -1
            PixelsPerStripe = 8 # Use a fairly high frequency pattern to separate the 0th from the 1st order
            bVertical = 0
            max_length = len(f"Gray: {NumDataPoints}")

            for region in range(0, NumRegions):
                print(f"Region: {region}")
                Variable = Reference
            
                for DataPoint in range(0, NumDataPoints):
                    output = f"Gray: {Variable}"
        
                    # Print the output with carriage return to overwrite the line
                    sys.stdout.write(f'\r{output.ljust(max_length)}')
                    sys.stdout.flush()

                    # Generate a stripe
                    image_lib.Generate_Stripe(Image.ctypes.data_as(POINTER(c_ubyte)), WFC.ctypes.data_as(POINTER(c_ubyte)), width, height, depth, Reference, Variable, PixelsPerStripe, bVertical, RGB, 0)
                    image_lib.Mask_Image(Image.ctypes.data_as(POINTER(c_ubyte)), width, height, depth, region, NumRegions, RGB)

                    # Decrement variable grayscale
                    Variable += StepBy

                    # Load the image to the SLM. Write image returns when the DMA is complete, and Image Write Complete
                    # Returns when the hardware memory bank is available to receive the next DMA
                    retVal = slm_lib.Write_image(board_number, Image.ctypes.data_as(POINTER(c_ubyte)), timeout_ms)
                    if(retVal != 1):
                        print ("DMA Failed")
                        break
                    else:
                        # Check the buffer is ready to receive the next image
                        retVal = slm_lib.ImageWriteComplete(board_number, timeout_ms)
                        if(retVal != 1):
                            print ("ImageWriteComplete failed, trigger never received?")
                            break

                    # Give the LC time to settle into the image
                    sleep(0.5)

                    # YOU FILL IN HERE...FIRST: read from your specific AI board, note it might help to clean up noise to average several readings
                    # SECOND: store the measurement in your AI_Intensities array
                    # AI_Intensities[Gray] = measured value
                print("\n")
                FileName = f"Raw{region}.csv"
                with open(FileName, mode='w', newline='') as file:
                    writer = csv.writer(file)  
                    # Write data to the CSV file
                    for i in range(len(AI_Intensities)):
                        writer.writerow([i, AI_Intensities[i]])
    retVal = slm_lib.Write_image(board_number, Blank.ctypes.data_as(POINTER(c_ubyte)), timeout_ms)
    if(retVal != 1):
        print ("DMA Failed")
    slm_lib.Delete_SDK()
