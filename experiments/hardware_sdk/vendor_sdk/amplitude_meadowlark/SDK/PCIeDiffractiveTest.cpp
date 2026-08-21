// Blink_SDK_example.cpp : Defines the entry point for the console application.
//


#include "Blink_C_wrapper.h"  // Relative path to SDK header.
#include "ImageGen.h"
#include <afx.h>
#include <afxwin.h> 

// ------------------------- Blink_SDK_example --------------------------------
// Simple example using the Blink_SDK DLL to load diffraction patterns to the SLM
// and measure the 1st order intensity by reading from an analog input board, then
// saving the measurements to raw files for post processing to generate either 
// a regional or global LUT.
// ----------------------------------------------------------------------------
int main()
{
	CStdioFile sFile;
	CString FileName, str;
	int board_number;
	unsigned int n_boards_found = 0U;
	int  constructed_okay = true;
	bool ExternalTrigger = false;
	bool FlipImmediate = false; //only supported on the 1k
	bool OutputPulseImageFlip = false;
  bool bSuccess;
  int timeout = 1000; //units in ms

	//call the construtor. If constructed okay returns 1, then hardware was found and communication was opened. 
	//Otherwise, the could be no hardware present, or another program could have the 
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
		board_number = 1;
		int NumDataPoints = 256; //8 bit controller
		int NumRegions = 1; //for a global LUT Num Regions = 1, for a regional LUT Num Regions = 64
		bool RGB = false; //PCIe interface never uses RGB images
		int height = Get_image_height(board_number);
		int width = Get_image_width(board_number);
		int depth = Get_image_depth(board_number); //bits per pixel
		int Bytes = depth / 8;
		int ImgSize = height * width * Bytes;
		
		SetWaitForTrigger(board_number, ExternalTrigger);
		SetFlipImmediate(board_number, FlipImmediate);
		SetOutputPulse(board_number, OutputPulseImageFlip);

		//when you are calibrating you want to load a linear LUT. If you are checking a calibration, load your calibrated LUT
		if (width == 1920)
			bSuccess = Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1920x1152_linearVoltage.LUT");
		if (width == 1024)
			bSuccess = Load_LUT_file(board_number, "C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1024x1024_linearVoltage.LUT");

    if (!bSuccess)
      printf("Load LUT File failed \n");

	//when calibrating a LUT **ALWAYS** leave the WFC blank
	unsigned char* WFC = new unsigned char[ImgSize];
	memset(WFC, 0, ImgSize);

	//Create arrays to hold the image data
	unsigned char* Image = new unsigned char[ImgSize];
	unsigned char* Blank = new unsigned char[ImgSize];
	memset(Image, 0, ImgSize);
	memset(Blank, 0, ImgSize);

	//Create an array to hold measurements from the analog input (AI) board
	float* AI_Intensities = new float[NumDataPoints];
	memset(AI_Intensities, 0, NumDataPoints);

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

		//load diffraction patterns, and record the 1st order intensity
		// ************* SET YOUR VARIABLES BELOW ACCORDINGLY ****************
		//1920x1152 8 in/12 out: reference should be 0, increment variable from 0 to 255 with StepBy = +1
		//1024x1024 8 in/12 out: reference should be 255, decrement variable from 255 down to 0 with StepBy = -1
		int Reference = 255;
		int Variable = 255;
		int StepBy = -1;
		int PixelsPerStripe = 64; //use a fairly high frequency pattern to separate the 0th from the 1st order
		int bVertical = 0;
		for (int region = 0; region < NumRegions; region++)
		{
			printf("Region: %d\n", region);

			for (int DataPoint = 0; DataPoint < NumDataPoints; DataPoint++)
			{
				printf("DataPoint: %d\r", DataPoint);

				//generate a stripe
				Generate_Stripe(Image, WFC, width, height, depth, Reference, Variable, PixelsPerStripe, bVertical, RGB);
				Mask_Image(Image, width, height, depth, region, NumRegions, RGB);

				//decrement variable grayscale
				Variable = Variable + StepBy;

				//load the iamge to the SLM. Write image returns when the DMA is complete, and Image Write Complete
				//returns when the hardware memory bank is available to receive the next DMA
				bSuccess = Write_image(board_number, Image, timeout);
        if(bSuccess)
        {
				  bSuccess = ImageWriteComplete(board_number, timeout);
          if (!bSuccess)
            printf("Image Write Complete failed\n");
        }
        else
          printf("Write Image failed\n");

				//give the LC time to settle into the image
				Sleep(10);

				//YOU FILL IN HERE...FIRST: read from your specific AI board, note it might help to clean up noise to average several readings
				//SECOND: store the measurement in your AI_Intensities array
				//AI_Intensities[Gray] = measured value

			}
			printf("\n");

			FileName.Format(_T("Raw%d.csv"), region);
			sFile.Open(FileName, CFile::modeCreate | CFile::modeWrite | CFile::typeText);
			for (int i = 0; i < NumDataPoints; i++)
			{
				str.Format(_T("%d, %f\n"), i, AI_Intensities[i]);
				sFile.WriteString(str);
			}
			sFile.Close();
		}
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


		delete[]WFC;
		delete[]Blank;
		delete[]Image;
		delete[]AI_Intensities;
		Delete_SDK();
		return EXIT_SUCCESS;
	}

	return EXIT_FAILURE;
}
