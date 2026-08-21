#include "Blink_C_wrapper.h"  // Relative path to SDK header.
#include "ImageGen.h"
#include "math.h"
#include <afxwin.h> 

// ------------------------- Blink_SDK_example --------------------------------
// Simple example using the Blink_SDK DLL to send a sequence of phase targets
// to a single SLM.
// To run the example, ensure that Blink_SDK.dll is in the same directory as
// the Blink_SDK_example.exe.
// ----------------------------------------------------------------------------
int main()
{
  int board_number;
  unsigned int n_boards_found = 0U;
  int  constructed_okay = true;
  bool ExternalTrigger = false; //hold off an image load until the SLM hardware receives an external trigger
  bool FlipImmediate = false; //only supported on the 1k
  bool OutputPulseImageFlip = false; //generate an output pulse when a new image begins loading on the SLM
  bool bSuccess;
  
  //This will call the constructor of the SDK
  Create_SDK(&n_boards_found, &constructed_okay);

  // Constructed_okay = 1 means success. If constructed okay is 0, then check to see the error. It could be that no
  // SLM is attached. This is acceptable, the software will allow the user to run in simulation mode. Or, 
  // it could mean that the driver handle is already open by another program (i.e. Blink or the Cal Kit) or that 
  // there is a problem with the device driver. 
  if (constructed_okay != 1)
    ::AfxMessageBox(Get_last_error_message());

  // this is the number of boards found. If more than one board is found, the software will allow you to interact with
  // each board individually through the board number. If no SLM is found, and you are running in simulation mode
  // then the num boards found will still be 1. 
  if (n_boards_found > 0)
  {
	  //get some specs on the attached SLM
	  board_number = 1;
	  int height = Get_image_height(board_number);
	  int width = Get_image_width(board_number);
	  int depth = Get_image_depth(board_number); //bits per pixel
	  int Bytes = depth/8;
	  int ImgSize = height*width*Bytes;
	  int timeout = 5000; //units in ms
	  
	  SetWaitForTrigger(board_number, ExternalTrigger);
	  SetFlipImmediate(board_number, FlipImmediate);
	  SetOutputPulse(board_number, OutputPulseImageFlip);
	
      //***you should replace *_linearVoltage.LUT with your custom LUT file***
	  //but for now open a generic LUT that linearly maps input graylevels to output voltages
	  //***Using *_linearVoltage.LUT does NOT give a linear phase response*** 
	  if(width == 1920)
		  bSuccess = Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1920x1152_linearVoltage.LUT");
	  if(width == 1024)
		  bSuccess = Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1024x1024_linearVoltage.LUT");
	
	  if(!bSuccess)
		  printf("Load LUT File failed\n");

	  //to keep the example generic a blank wavefront correction is used, you can replace this with your real wavefront correction
	  unsigned char* WFC = new unsigned char[ImgSize];
	  memset(WFC, 0, ImgSize);

	  // Create two vectors to hold values for two SLM images
	  unsigned char* Blank = new unsigned char[ImgSize];
	  unsigned char* ImageOne = new unsigned char[ImgSize];
	  unsigned char* ImageTwo = new unsigned char[ImgSize];
	  memset(Blank, 0, ImgSize);
	  memset(ImageOne, 0, ImgSize);
	  memset(ImageTwo, 0, ImgSize);
	
	  //start the SLM with a blank image
	  bSuccess = Write_image(board_number, Blank, timeout);
	  if (bSuccess)
	  {
		  bSuccess = ImageWriteComplete(board_number, timeout);
		  if (!bSuccess)
			  printf("Image Write Complete failed\n");
	  }
	  else
		  printf("Write Image failed\n");
	
	  // Generate phase gradients
	  int VortexCharge = 5;
	  bool RGB = false;
	  bool fork = false;
	  Generate_LG(ImageOne, WFC, width, height, depth, VortexCharge, width / 2.0, height / 2.0, fork, RGB);
	  VortexCharge = 3;
	  Generate_LG(ImageTwo, WFC, width, height, depth, VortexCharge, width / 2.0, height / 2.0, fork, RGB);

	  for (int i = 0; i < 10; i++)
	  {
		  //write image returns on DMA complete, ImageWriteComplete returns when the hardware
		  //image buffer is ready to receive the next image. Breaking this into two functions is 
		  //useful for external triggers. It is safe to apply a trigger when Write_image is complete
		  //and it is safe to write a new image when ImageWriteComplete returns
		  bSuccess = Write_image(board_number, ImageOne, timeout);
		  if(bSuccess)
		  {
			  bSuccess = ImageWriteComplete(board_number, timeout);
        if(!bSuccess)
          ::AfxMessageBox(Get_last_error_message());

			  Sleep(500); //if using external triggers omit sleep commands
		  }
		  else 
        ::AfxMessageBox(Get_last_error_message());

		  bSuccess = Write_image(board_number, ImageTwo, timeout);
      if(bSuccess)
      {
		    bSuccess = ImageWriteComplete(board_number, timeout);
        if(!bSuccess)
          ::AfxMessageBox(Get_last_error_message());

		    Sleep(500); //if using external triggers omit sleep commands
      }
      else
        ::AfxMessageBox(Get_last_error_message());
	  }
	  //end with a blank image
	  bSuccess = Write_image(board_number, Blank, timeout);
	  if (bSuccess)
	  {
		  bSuccess = ImageWriteComplete(board_number, timeout);
		  if (!bSuccess)
			  printf("Image Write Complete failed\n");
	  }
	  else
		  printf("Write Image failed\n");

	  delete[]Blank;
	  delete[]ImageOne;
	  delete[]ImageTwo;
	  delete[]WFC;

	  Delete_SDK();
	  return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}