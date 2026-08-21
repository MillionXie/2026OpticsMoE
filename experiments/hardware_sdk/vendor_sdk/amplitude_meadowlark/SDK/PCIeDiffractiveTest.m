% Example usage of Blink_C_wrapper.dll
% Meadowlark Optics Spatial Light Modulators
% last updated: August 1 2024

% Load the DLL
% Blink_C_wrapper.dll. If you move the example, take DLLs from the SDK folder
% to the new folder such that all dependent DLLs can be found
if ~libisloaded('Blink_C_wrapper')
    loadlibrary('Blink_C_wrapper.dll', 'Blink_C_wrapper.h');
end

% This loads the image generation functions
if ~libisloaded('ImageGen')
    loadlibrary('ImageGen.dll', 'ImageGen.h');
end

% Basic parameters for calling Create_SDK
num_boards_found = libpointer('uint32Ptr', 0);
constructed_okay = libpointer('int32Ptr', 0);
wait_For_Trigger = 0; % This feature turns on or off listening for an external trigger, use 1 for 'on' or 0 for 'off'
flip_immediate = 0; % Only supported on the 1024, interrupts an image refresh to start loading new data
OutputPulseImageFlip = 0; % This feature enables the hardware to generate an output pulse when a new image begins loading to the SLM
timeout_ms = 5000;
RGB = 0;

% This will call the constructor of the SDK
calllib('Blink_C_wrapper', 'Create_SDK', num_boards_found, constructed_okay);

% Constructed_okay = 1 means success. If constructed okay is 0, then check to see the error. It could be that no
% SLM is attached. This is acceptable, the software will allow the user to run in simulation mode. Or, 
% it could mean that the driver handle is already open by another program (i.e. Blink or the Cal Kit) or that 
% there is a problem with the device driver. 
if constructed_okay.value ~= 1  
    disp(calllib('Blink_C_wrapper', 'Get_last_error_message'));
end

% this is the number of boards found. If more than one board is found, the software will allow you to interact with
% each board individually through the board number. If no SLM is found, and you are running in simulation mode
% then the num boards found will still be 1. 
if num_boards_found.value > 0  
    board_number = 1;
    disp('Blink SDK was successfully constructed');
    fprintf('Found %u SLM controller(s)\n', num_boards_found.value);
    
	% set some dimensions
	height = calllib('Blink_C_wrapper', 'Get_image_height', board_number);
    width = calllib('Blink_C_wrapper', 'Get_image_width', board_number);
	depth = calllib('Blink_C_wrapper', 'Get_image_depth', board_number); %bits per pixel
	Bytes = depth/8;
    NumDataPoints = 256;
    NumRegions = 1;
    
    calllib('Blink_C_wrapper', 'SetWaitForTrigger', board_number, wait_For_Trigger);
	calllib('Blink_C_wrapper', 'SetFlipImmediate', board_number, flip_immediate);
	calllib('Blink_C_wrapper', 'SetOutputPulse', board_number, OutputPulseImageFlip);
	
    % To measure the raw optical response we want to linearly increment the voltage on the pixels by using a linear LUT
    if width == 1920
		load_lut_status = calllib('Blink_C_wrapper', 'Load_LUT_file', board_number, 'C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1920x1152_linearVoltage.LUT');
    end
    if width == 1024
		load_lut_status = calllib('Blink_C_wrapper', 'Load_LUT_file', board_number, 'C:\\Program Files\\Meadowlark Optics\\Blink Plus\\LUT Files\\1024x1024_linearVoltage.LUT');
    end

    % load_lut_status == 1 if successful, load_lut_status == 0 if unsuccessful
    if load_lut_status == 0
        disp('Error loading LUT file. Check LUT file and path.')
    end

    % allocate arrays for our images
    Blank = libpointer('uint8Ptr', zeros(width*height*Bytes,1));
    Image = libpointer('uint8Ptr', zeros(width*height*Bytes,1));
	
	% ***ALWAYS*** use a blank wavefront correction when calibrating a LUT
	WFC = libpointer('uint8Ptr', zeros(width*height*Bytes,1));

    % Create an array to hold measurements from the analog input (AI) board
    AI_Intensities = zeros(NumDataPoints,2);
    
	% begin with the SLM blank
    write_image_status = calllib('Blink_C_wrapper', 'Write_image', board_number, Blank, timeout_ms);
    if write_image_status == 1
        image_write_complete_status = calllib('Blink_C_wrapper', 'ImageWriteComplete', board_number, timeout_ms);
        if image_write_complete_status == 0
            disp('ImageWriteComplete failed, trigger never received?');
        end
    else
        disp('The call to Write_image was unsuccessful.');
    end
    % Use a high frequency grating to separate the 0th and 1st orders, a
    % period of 8 is generally good
    PixelsPerStripe = 8;
    
    %When calibrating you write a series of stripes to the SLM. Use the
    %summary below to set the reference, the variable grayscale, and the
    %value you step the variable grayscale by.
    
    %1920x1152, reference = 0, variable = 0 to 255 in steps of +1
    %1024x1024, reference = 255, variable = 255 to 0 in steps of -1
    Reference = 255;
    Variable = 255;
    StepBy = -1;
    bVertical = 0;
    %loop through each region
    for Region = 0:(NumRegions-1)
        fprintf('Region: %d\n', Region);
        Variable = Reference;
        AI_Index = 1;
        %loop through each graylevel
        for TestPoint = 0:(NumDataPoints-1)
            output = sprintf('Gray: %d', Variable);
            MaxLength = strlength(output);
            % Print the output with carriage return to overwrite the line
            fprintf('%s', output);
            
            pause(0.2);
            %Generate the stripe pattern and mask out current region
            calllib('ImageGen', 'Generate_Stripe', Image, WFC, width, height, depth, Reference, Variable, PixelsPerStripe, bVertical, RGB);
            calllib('ImageGen', 'Mask_Image', Image, width, height, depth, Region, NumRegions, RGB);
            
            %Step the variable grayscale
            Variable = Variable + StepBy;
            
            %write the image
            
            write_image_status = calllib('Blink_C_wrapper', 'Write_image', board_number, Image, timeout_ms);
            if write_image_status == 1
                image_write_complete_status = calllib('Blink_C_wrapper', 'ImageWriteComplete', board_number, timeout_ms);
                if image_write_complete_status == 0
                    disp('ImageWriteComplete failed, trigger never received?');
                end
            else
                disp('The call to Write_image was unsuccessful.');
            end
            %let the SLM settle for 10 ms
            pause(0.01);
            
            %YOU FILL IN HERE...FIRST: read from your specific AI board, note it might help to clean up noise to average several readings
            %SECOND: store the measurement in your AI_Intensities array
            AI_Intensities(AI_Index, 1) = TestPoint; %This is the varable graylevel you wrote to collect this data point
            AI_Intensities(AI_Index, 2) = 0; % HERE YOU NEED TO REPLACE 0 with YOUR MEASURED VALUE FROM YOUR ANALOG INPUT BOARD
 
            AI_Index = AI_Index + 1;
            fprintf(repmat('\b', 1, MaxLength));
        end
        
        % dump the AI measurements to a csv file
        filename = ['Raw' num2str(Region) '.csv'];
        csvwrite(filename, AI_Intensities);  
    end
	% End with a blank image
    write_image_status = calllib('Blink_C_wrapper', 'Write_image', board_number, Blank, timeout_ms);
    if write_image_status == 1
        image_write_complete_status = calllib('Blink_C_wrapper', 'ImageWriteComplete', board_number, timeout_ms);
        if image_write_complete_status == 0
            disp('ImageWriteComplete failed, trigger never received?');
        end
    end
    % Always call Delete_SDK before exiting
    calllib('Blink_C_wrapper', 'Delete_SDK');
end

%destruct
if libisloaded('Blink_C_wrapper')
    unloadlibrary('Blink_C_wrapper');
end

if libisloaded('ImageGen')
    unloadlibrary('ImageGen');
end