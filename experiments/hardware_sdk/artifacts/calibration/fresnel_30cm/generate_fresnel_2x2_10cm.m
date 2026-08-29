close all; clear;

% Current phase SLM contract.
slm_width = 1920;
slm_height = 1200;
phase_center_x = 980.0;
phase_center_y = 590.0;

% 10 cm Fresnel-array contract.
array_rows = 2;
array_cols = 2;
window = 350;
f = 10e-2;
pixel_pitch = 8e-6;
lambda = 532e-9;
k = 2*pi/lambda;

if mod(window, 2) ~= 0
    error('window must be even for an exactly centered 2x2 array');
end
if phase_center_x ~= round(phase_center_x) || phase_center_y ~= round(phase_center_y)
    error('This exported mask requires integer edge-coordinate center_x/center_y');
end

% Symmetric pixel-center coordinates for an even 350x350 tile.
coordinate = ((1:window) - 0.5 - window/2) * pixel_pitch;
[X, Y] = meshgrid(coordinate, coordinate);
tile_phase = -k * (X.^2 + Y.^2) / (2*f);

masks = zeros(slm_height, slm_width);

% Complete 700x700 footprint centered at edge coordinate (980,590).
left = round(phase_center_x - array_cols*window/2) + 1;
top = round(phase_center_y - array_rows*window/2) + 1;
right = left + array_cols*window - 1;
bottom = top + array_rows*window - 1;

if left < 1 || top < 1 || right > slm_width || bottom > slm_height
    error('The Fresnel array footprint exceeds the phase SLM canvas');
end

for row = 1:array_rows
    for col = 1:array_cols
        row_start = top + (row-1)*window;
        row_end = row_start + window - 1;
        col_start = left + (col-1)*window;
        col_end = col_start + window - 1;
        masks(row_start:row_end, col_start:col_end) = tile_phase;
    end
end

masks = mod(masks, 2*pi);
phase_uint8 = uint8(round(masks/(2*pi)*255));

output_dir = fileparts(mfilename('fullpath'));
output_name = 'fresnel_2x2_f100mm_w350_center980_590_1920x1200.bmp';
imwrite(phase_uint8, fullfile(output_dir, output_name), 'bmp');

fprintf('Saved %s\n', fullfile(output_dir, output_name));
fprintf('SLM: %dx%d, pitch %.1f um, lambda %.0f nm, f %.0f mm\n', ...
    slm_width, slm_height, pixel_pitch*1e6, lambda*1e9, f*1e3);
fprintf('Array footprint: x=%d:%d, y=%d:%d (MATLAB 1-based)\n', ...
    left, right, top, bottom);

