close all; clear;
masks = zeros([1152,1920]);
for i = 1:2
    for j = 1:2
        window = 350;
        [X,Y] = meshgrid(1:window,1:window);
        center_point = window/2.0;
        f = 30*1e-2;
        ps = 9.2*1e-6;
        lambda = 532*1e-9;
        k = 2*pi/lambda;
        middle = -k*((ps*(X-center_point)).^2+ (ps*(Y-center_point)).^2)/(2*f);

        save_mask_phase = middle;
        
        offset_x = (1.5-i)*350;
        offset_y = (1.5-j)*350;
        masks(end/2-window/2+1-offset_x:end/2+window/2-offset_x,end/2-window/2+1-offset_y:end/2+window/2-offset_y) = save_mask_phase;

    end
end
        masks = mod(masks,2*pi);
        name = '\30cm_92_532nm_array.bmp';
        imwrite(masks/2/pi,name);