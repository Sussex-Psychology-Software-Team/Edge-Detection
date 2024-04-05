%https://numpy.org/devdocs/user/numpy-for-matlab-users.html
%note requires Image Processing Toolbox
addpath('MATLAB')
addpath('export')
addpath('images')
addpath('plots')
addpath('text')
addpath('csvs')

format long %output decimal places same as python
%digitsOld = digits(32); %note consider using vpa() function to increase precision e.g. vpa(pi)

FILTER_SIZE = 31;
GABOR_BINS = 24;
BINS_VEC = linspace(0, 2*pi, GABOR_BINS+1)';
BINS_VEC = BINS_VEC(1:end-1);
CIRC_BINS = 48;
IMAGE_DIR = 'images';
PLOT_DIR = 'plots';
TEXT_DIR = 'text';
EXPORT_DIR = 'export';
TIMESTAMP = datestr(now, 'yyyy.mm.dd-HHMMSS-dddd');

MAX_PIXELS = 300*400;
MAX_DIAGONAL = 500;
MAX_TASKS = 0; % if MAX_TASKS==0, no forking

filter_bank = filterBank(GABOR_BINS, FILTER_SIZE);
file_list = get_file_list(IMAGE_DIR, '.png');

for i = 1:(filter_bank.num_filters-1)
    theta = BINS_VEC(i);
    filter_bank = filter_bank.set_flt(i, filterBank.create_gabor(FILTER_SIZE, theta, 3,i));
end

log_file = fullfile(EXPORT_DIR, ['MATLAB_export-' TIMESTAMP '.csv']);

RANGES = [20 80; 80 160; 160 240];
result_names = char(strcat('avg-shannon',num2str(RANGES(:,1)),'-',num2str(RANGES(:,2))), strcat('avg-shannon-nan',num2str(RANGES(:,1)),'-',num2str(RANGES(:,2))),'first_order-shannon', 'edge-density');

fid = fopen(log_file, 'w');
fprintf(fid, 'image,%s\n', strjoin(cellstr(result_names), ','));

i = 1;
is_child = false;
while i <= length(file_list)
    if is_child || MAX_TASKS == 0
        img = filterImage(file_list{i}, MAX_PIXELS);
        %imwrite(uint8(img.image_raw),['plots/MATLAB_imraw_' i '.png'])

        img = img.run_filterbank(filter_bank);
        %writematrix(img.image_raw, ['MATLAB_imraw_' num2str(i) '.csv']); %raw image values csv
        [counts, complex_before] = do_counting(img, file_list{i});
        [normalized_counts, circular_mean_angle, circular_mean_length, shannon, shannon_nan] = do_statistics(counts);
        results = [];
        for r = RANGES
            results = [results, mean(mean(shannon(r(1):r(2), :), 1))];
            results = [results, mean(mean(shannon_nan(r(1):r(2), :), 1,"omitnan"),"omitnan")];
        end
        results = [results, do_first_order(img), complex_before];
    
        dlmwrite(fullfile(TEXT_DIR, ['MATLAB_',file_list{i} '_shannon.txt']), mean(shannon, 1), 'precision', 8);
        dlmwrite(fullfile(TEXT_DIR, ['MATLAB_',file_list{i} '_shannon-nan.txt']), mean(shannon_nan, 1, 'omitnan'), 'precision', 8);
    
        fprintf(fid, '%s,%s\n', file_list{i}, strjoin(cellfun(@num2str, num2cell(results), 'UniformOutput', false), ','));
        i = i + 1;
        if MAX_TASKS ~= 0 && (mod(i, MAX_TASKS) == 0 || i == length(file_list))
            fprintf('child: I''m leaving\n');
            exit;
        end
    else
        child_pid = system(['nohup ', mfilename, ' &']);
        if child_pid == 0
            fprintf('child: I''m doing work, my pid is %d\n', getpid());
            is_child = true;
        else
            [pid, status] = system(['wait ', num2str(child_pid)]);
            fprintf('parent: wait returned, pid = %d, status = %d\n', pid, status);
            if status ~= 0
                fprintf('parent: child terminated with an error, parent quitting.\n');
                exit(1);
            end
            i = i + MAX_TASKS;
        end
    end
end

fclose(fid);

% Function to get a list of files in a directory
function file_list = get_file_list(directory, extension)
    files = dir(fullfile(directory, ['*' extension]));
    file_list = {files.name};
end

function [counts, complex_before] = do_counting(filter_img, filename)
    % Temporary solution as MATLAB doesn't like globals
    MAX_DIAGONAL = 500;
    GABOR_BINS = 24;
    CIRC_BINS = 48;
    % Creates histogram (distance, relative orientation in image, relative gradient)
   
    % Cutoff minor filter responses
    normalize_fac = numel(filter_img.resp_val);
    complex_before = sum(filter_img.resp_val(:)) / normalize_fac;
    sorted_values = sort(filter_img.resp_val(:));
    cutoff_value = sorted_values(end - 9999);
    filter_img.resp_val(filter_img.resp_val < cutoff_value) = 0;

    % Visualize remaining responses
    [h, w] = size(filter_img.resp_val);
    a = zeros(h, w);

    [ey, ex] = find(filter_img.resp_val);
    a(sub2ind([h, w], ey, ex)) = filter_img.resp_val(sub2ind([h, w], ey, ex));
    writematrix(a, ['csvs/MATLAB_' filename '.csv']); %specific values - zoom out on excel to see picutre in numbers
    figure(1);

    imshow(a, 'DisplayRange', [], 'Colormap', gray, 'Interpolation', 'nearest', 'InitialMagnification', 'fit'); %[0 max(filter_img.resp_val(:))]
    title(['MATLAB_edges_', filename]);
    saveas(gcf, fullfile('plots', ['MATLAB_edges_', filename]), 'png');
    close(gcf);

    % Lookup tables to speed up calculations
    [xx, yy] = meshgrid(linspace(-w, w, 2*w+1), linspace(-h, h, 2*h+1));
    dist = sqrt(xx.^2 + yy.^2);

    indices = sub2ind(size(filter_img.resp_bin), ey, ex);
    orientations = filter_img.resp_bin(indices);

    counts = zeros(MAX_DIAGONAL, CIRC_BINS, GABOR_BINS);

    disp(['Counting ', filter_img.image_name, ' ', num2str(filter_img.image_size()), ' comparing ', num2str(numel(ex))]);

    for cp = 1:numel(ex)
        orientations_rel = mod(orientations - orientations(cp) + GABOR_BINS, GABOR_BINS);

        lin_idx = sub2ind(size(dist), (ey - ey(cp)) + h, (ex - ex(cp)) + w);
        distance_rel = round(dist(lin_idx));
        distance_rel(distance_rel >= MAX_DIAGONAL) = MAX_DIAGONAL - 1;
        
        direction = mod(round(atan2(ey-ey(cp), ex-ex(cp)) / (2.0*pi)*CIRC_BINS + (orientations(cp)/GABOR_BINS*CIRC_BINS)), CIRC_BINS);

        % Compute the product to be added
        e_idx = sub2ind(size(filter_img.resp_val), ey(:), ex(:));
        cp_idx = sub2ind(size(filter_img.resp_val), ey(cp), ex(cp));
        contribution = filter_img.resp_val(e_idx) .* filter_img.resp_val(cp_idx);
        
        % create linear indices (where to add values to counts)
        linear_indices = sub2ind(size(counts), distance_rel+1, direction+1, orientations_rel+1);
        counts(linear_indices) = contribution(:);
    end
end

function H = entropy(a)
    if sum(a) ~= 1.0 && sum(a) > 0
        a = a / sum(a);
    end
    v = a > 0.0;
    H = -sum(a(v) .* log2(a(v)));
end

function [normalized_counts, circular_mean_angle, circular_mean_length, shannon, shannon_nan] = do_statistics(counts)
    % TEMP GLOBALS
    GABOR_BINS = 24;
    BINS_VEC = linspace(0, 2*pi, GABOR_BINS+1); BINS_VEC = BINS_VEC(1:end-1);
    BINS_VEC = permute(BINS_VEC, [1, 3, 2]); %add dimension for element-wise multiplication
    % Normalize counts by sum
    counts_sum = sum(counts, 3) + 0.00001;
    normalized_counts = counts ./ repmat(counts_sum, [1, 1, size(counts, 3)]);

    x = normalized_counts .* cos(BINS_VEC);
    y = normalized_counts .* sin(BINS_VEC);

    mean_vector = mean(x + 1i * y, 3);
    circular_mean_angle = mod(angle(mean_vector) + 2 * pi, 2 * pi);
    circular_mean_length = abs(mean_vector);

    % Correction as proposed by Zar 1999
    d = 2 * pi / GABOR_BINS;
    c = d / (2.0 * sin(d / 2.0));
    circular_mean_length = circular_mean_length * c;

    [d, a, ~] = size(normalized_counts);
    shannon = zeros(d, a);
    shannon_nan = zeros(d, a);

    for di = 1:d
        for ai = 1:a
            shannon(di, ai) = entropy(normalized_counts(di, ai, :));
            if counts_sum(di, ai) > 1
                shannon_nan(di, ai) = shannon(di, ai);
            else
                shannon_nan(di, ai) = NaN;
            end
        end
    end
end

function first_order = do_first_order(img)
    GABOR_BINS = 24;
    first_order_bin = zeros(1, GABOR_BINS);
    for b = 1:GABOR_BINS
        first_order_bin(b) = sum(img.resp_val(img.resp_bin == b));
    end
    first_order = entropy(first_order_bin);
end
