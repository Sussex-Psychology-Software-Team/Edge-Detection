classdef filterBank
    properties
        num_filters
        flt_size
        flt_raw
    end
    
    methods
        function obj = filterBank(num_filters, flt_size)
            obj.num_filters = num_filters;
            obj.flt_size = flt_size;
            obj.flt_raw = zeros(num_filters, flt_size, flt_size);
        end

        function obj = set_flt(obj, n, flt)
            if size(flt) ~= [obj.flt_size, obj.flt_size]
                error(['wrong filter size, expected ' num2str(obj.flt_size) ' got ' num2str(size(flt))]);
            elseif n >= obj.num_filters
                error(['filternumber ' num2str(n) ' is out of range']);
            else
                % store the raw filter
                obj.flt_raw(n, :, :) = cast(flt, 'like', obj.flt_raw);
            end
        end

        function show(obj, n) %not used
            figure(1); 
            title(['filter ' num2str(n) ' raw']);
            imshow(squeeze(obj.flt_raw(n,:,:)), 'Colormap', gray,'Interpolation', 'nearest');
        end

    end

    methods(Static)
        function gaussian = create_gaussian(size, sigma) %note: not currently in use
            valsy = linspace(-size/2+1, size/2, size);
            valsx = linspace(-size/2+1, size/2, size);
            [xgr, ygr] = meshgrid(valsx, valsy);
            gaussian = exp(-(xgr.^2 + ygr.^2)/(2*sigma^2));
            gaussian = gaussian/sum(gaussian(:));
        end

        function gabor = create_gabor(size, theta, octave,i)
            amplitude = 1.0;
            phase = pi/2.0;
            frequency = 0.5^octave;
            hrsf = 4;
            sigma = 1/(pi*frequency) * sqrt(log(2)/2) * (2.0^hrsf+1)/(2.0^hrsf-1);
            valsy = linspace(-size/2+1, size/2, size);
            valsx = linspace(-size/2+1, size/2, size);
            [xgr, ygr] = meshgrid(valsx, valsy);

            omega = 2*pi*frequency;
            gaussian = exp(-(xgr.^2 + ygr.^2)/(2*sigma^2));
            
            slant = xgr.*(omega*sin(theta)) + ygr.*(omega*cos(theta));
            
            gabor = gaussian .* amplitude.*cos(slant + phase);
        end
    end
end