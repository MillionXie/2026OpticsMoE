// BlinkSdkExample.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include "Blink_C_wrapper.h"
#include "ImageGen.h"

int _tmain(int argc, _TCHAR* argv[])
{
  FreeConsole();

  Create_SDK();

  //Read the height and width of the SLM found. The default is 1920x1152 if no SLM is found. The default location
  //is immediately right of the primary monitor. If you monitor is smaller than 1920x1152 you can still run, you will
  //just see a sub-section of the image on the monitor. 
  int height = Get_Height();
  int width = Get_Width();
  int depth = Get_Depth(); //will return 8 for 8-bit, or 10 for 10-bit
  int BytesPerImage = 4; //RGBA
  
  // Path to the calibration of voltage to phase. The name "1920x1152_linearVoltage.lut" is a bit confusing. 
  // This linearly maps input voltages to output voltages, thus removing any calibration. 
  // To be generic this example uses 1920x1152_linearVoltage.lut, but you should link to your custom LUT delivered 
  // with the SLM. 
  if(height == 1152)
	Load_lut("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\LUT Files\\1920x1152_linearVoltage.lut");
  else
  {
	  if(depth == 8)
		  Load_lut("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\LUT Files\\19x12_8bit_linearVoltage.lut");
	  if(depth == 10)
		  Load_lut("C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\LUT Files\\19x12_10bit_linearVoltage.lut");
  }

  // Create two vectors to hold image data, init as RGB images
  unsigned char* ImageOne = new unsigned char[width*height*BytesPerImage];
  memset(ImageOne, 0, width*height*BytesPerImage);
  unsigned char* ImageTwo = new unsigned char[width*height*BytesPerImage];
  memset(ImageTwo, 0, width*height * BytesPerImage);

  //wavefront correction - leave blank for now
  unsigned char* WFC = new unsigned char[width*height*BytesPerImage];
  memset(WFC, 0, width*height*BytesPerImage);

  // Generate phase gradients, the returned image is superimposed with the wavefront correction
  int VortexCharge = 5;
  bool RGB = true;
  Generate_LG(ImageOne, WFC, width, height, depth, VortexCharge, width / 2.0, height / 2.0, false, RGB);
  VortexCharge = 3;
  Generate_LG(ImageTwo, WFC, width, height, depth, VortexCharge, width / 2.0, height / 2.0, false, RGB);

  //indicate that our images are RGB images
  bool isEightBit = false;

  //load the images on the SLM in a loop
  for (int i = 0; i < 10; i++)
  {
    Write_image(ImageOne, isEightBit);
    Sleep(1000);
    Write_image(ImageTwo, isEightBit);
    Sleep(1000);
  }

  //clean up allocated arrays
  delete[] ImageOne;
  delete[] ImageTwo;

  // Destruct
  Delete_SDK();

  return 0;
}

