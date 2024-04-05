classdef filterImage
    properties
        image_name string
        image_raw double
        image_flt double
        resp_bin double
        resp_val double
    end
    
    methods
        function obj = filterImage(filename, max_pixels)
            obj.image_name = filename;
            img = imread(filename);

            % resize to max_pixels while keeping dimensions
            if max_pixels
                [h, w, z] = size(img);
                a = sqrt( max_pixels / (h*w) ); %rounded 1d.p. higher than python version
                img = imresize(img, a, Method='lanczos3'); % note resulting size is 1px bigger than Python implementation
                % On use of Lanczos3: https://github.com/python-pillow/Pillow/discussions/5519
                % https://www.reddit.com/r/learnpython/comments/uggbw1/pil_image_resize_lanczos_algorithm_i_need_to/
            end

            %obj.image_raw = rgb2gray(img); % built-in option. 
            % ITU-R 601-2 luma transform
            obj.image_raw = 0.299 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3);
            %mean(mean(obj.image_raw)) %slightly off, perhaps just extra pixel? fewer d.p.s on the luma constants?
        end

        function obj = run_filterbank(obj, filter_bank)
            [h, w] = size(obj.image_raw);
            num_filters = size(filter_bank.flt_raw, 1);
            obj.image_flt = zeros(num_filters, h, w);
            for i = 1:num_filters
                % note conv2 with 'same' 0-pads borders and leads to edges around image
                %obj.image_flt(i, :, :) = conv2(obj.image_raw, squeeze(filter_bank.flt_raw(i,:,:)), 'same');
                flt_sq = squeeze(filter_bank.flt_raw(i,:,:));

                % Reflection convolution by hand
                filter_size = size(flt_sq);
                if mod(filter_size, 2) == 0
                    % If it's even, adjust the pad_size to ensure "valid" convolution
                pad_size = (filter_size) / 2;
                else
                    pad_size = (filter_size - 1) / 2;
                end
                padded_image = padarray(obj.image_raw, pad_size, 'symmetric');
                obj.image_flt(i, :, :) = conv2(double(padded_image), double(flt_sq), 'valid');
                
                %Convolution with Image Processing Toolbox - could also use padded_image rather than obj.image_raw
                %obj.image_flt(i, :, :) = imfilter(obj.image_raw, flt_sq, 'conv', 'symmetric'); 
                
                %EXPORT INVIDUAL FILTER or CONVOLUTIONS:
                %b = squeeze(obj.image_flt(i,:,:)); % use filter_bank.flt_raw(i,:,:) or obj.image_flt(i, :, :)
                %[x,y] = size(b);
                %a = reshape(b, x, y);
                %writematrix(a, ['MATLAB_conv_' num2str(i) '.csv']);
                %imwrite(uint8(a),['MATLAB_conv_' num2str(i) '.png'])
            end
            
            [M,I] = max(obj.image_flt,[],1);
            obj.resp_bin = squeeze(I);
            obj.resp_val = squeeze(M);
        end

        function sz = image_size(obj)
            sz = size(obj.image_raw);
        end

        function show(obj)
            figure('Name', 'image_raw'); 
            title(obj.image_name);
            imshow(obj.image_raw, 'cmap', 'gray', 'interpolation', 'nearest');
            figure('Name', 'resp_val');
            imshow(squeeze(obj.resp_val), 'cmap', 'gray', 'interpolation', 'nearest');
            figure('Name', 'resp_bin');
            imshow(squeeze(obj.resp_bin), 'cmap', 'gray', 'interpolation', 'nearest');
            figure('Name', 'edges overlay');
            imshow(hsv2rgb(cat(3, double(squeeze(obj.resp_bin))/GABOR_BINS, double(squeeze(obj.resp_val))/max(obj.resp_val(:)), double(obj.image_raw)/255)), 'cmap', 'gray', 'interpolation', 'nearest');
        end
    end
end