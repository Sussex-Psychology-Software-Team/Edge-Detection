import matplotlib.pyplot as plt
import numpy as np
import os, sys, time # OS module in Python provides functions for interacting with the operating system. 
import warnings # Warning messages
import PIL
# PIL is used for image processing
from PIL import Image # import the image module from the pillow & call the image. 

from matplotlib.colors import hsv_to_rgb # the function is used to convert hsv values to rgb. 
# hsv: It is an array-like argument in the form of (…, 3) where all values are assumed to be in the range of 0 to 1

# Scipy package contains various functions for multidimensional image processing.
from scipy import ndimage # Multidimensional image processing (scipy.ndimage).

from shutil import rmtree # The shutil module offers a number of high-level operations on files and collections of files. 

# scikit-image is a collection of algorithms for image processing.
from skimage.draw import polygon # Generate coordinates of pixels inside a polygon.  
from skimage.draw import disk as circle # Generate coordinates of pixels within circle.
import importlib


np.set_printoptions(threshold=sys.maxsize)

FILTER_SIZE = 31
GABOR_BINS = 24
BINS_VEC = np.linspace(0, 2*np.pi, GABOR_BINS+1)[:-1]
CIRC_BINS = 48
IMAGE_DIR = "images"
PLOT_DIR = "plots"
TEXT_DIR = "text"
EXPORT_DIR = "export"

MAX_PIXELS = 300*400
MAX_DIAGONAL = 500

MAX_TASKS = 0 # if MAX_TASKS==0, no forking

#-------------------------------------------------------------------------------

class Filter_bank:
    def __init__(self, num_filters, flt_size):
        """set number of filters and the expected filter size in contructor"""
        self.num_filters = num_filters
        self.flt_size = flt_size
        self.flt_raw = np.zeros([num_filters, flt_size, flt_size])

    def set_flt(self, n, flt):
        """store filter n in filter array"""
        if flt.shape != (self.flt_size,self.flt_size):
            raise Exception("wrong filter size, expected "+str(self.flt_size)+" got "+str(flt.shape))
        elif n >= self.num_filters:
            raise Exception("filternumber "+str(n)+" is out of range")
        else:
            # store the raw filter
            #print(flt)
            self.flt_raw[n,:,:] = flt
            #print(np.mean(self.flt_raw[n,:,:]))

    def show(self, n):
        """show the raw filter and the frequencies"""
        plt.figure(1), plt.title("filter "+str(n)+" raw"), plt.imshow(self.flt_raw[n,:,:], cmap='gray', interpolation='nearest')
        plt.show()

    @staticmethod
    def create_gaussian(size, sigma=2):
        valsy = np.linspace(-size/2+1, size/2, size)
        valsx = np.linspace(-size/2+1, size/2, size)
        xgr, ygr = np.meshgrid(valsx, valsy)
        gaussian = np.exp(-(xgr**2 + ygr**2)/(2*sigma*sigma))
        gaussian = gaussian/np.sum(gaussian)
        return gaussian

    @staticmethod
    def create_gabor(size, theta=0, octave=3):
        amplitude = 1.0
        phase = np.pi/2.0
        frequency = 0.5**octave # 0.5**(octave+0.5)
        hrsf = 4 # half response spatial frequency bandwith
        sigma = 1/(np.pi*frequency) * np.sqrt(np.log(2)/2) * (2.0**hrsf+1)/(2.0**hrsf-1)
        valsy = np.linspace(-size/2+1, size/2, size)
        valsx = np.linspace(-size/2+1, size/2, size)
        xgr,ygr = np.meshgrid(valsx, valsy)

        omega = 2*np.pi*frequency
        gaussian = np.exp(-(xgr*xgr + ygr*ygr)/(2*sigma*sigma))
        slant = xgr*(omega*np.sin(theta)) + ygr*(omega*np.cos(theta))
        gabor = gaussian * amplitude*np.cos(slant + phase)
        # e^(-(x^2+y^2)/(2*1.699^2)) *cos(pi/4*(x*sin(2)+y*cos(2)) + pi/2)

        return gabor

#-------------------------------------------------------------------------------

class FilterImage:
    def __init__(self, filename, max_pixels=None):
        self.image_name = filename
        img = Image.open(filename)
        # resize to max_pixels while keeping dimensions
        if max_pixels!=None:
            a = np.sqrt(max_pixels / float(img.size[0]*img.size[1]))
            img = img.resize((int(img.size[0]*a),int(img.size[1]*a)), PIL.Image.Resampling.LANCZOS)
        self.image_raw = np.asarray(img.convert('L'), dtype='float'); # using ITU-R 601-2 luma transform
        
    def run_filterbank(self, filter_bank):
        (h, w) = self.image_raw.shape
        num_filters = filter_bank.flt_raw.shape[0]
        #print(np.mean(filter_bank.flt_raw,axis=tuple([0,1,2])))
        self.image_flt = np.zeros((num_filters,h,w))
        #print(np.mean(filter_bank.flt_raw[1,:,:],axis=tuple([0,1])))
        for i in range(num_filters):
            self.image_flt[i,:,:] = ndimage.convolve(self.image_raw, filter_bank.flt_raw[i,:,:])
            # EXPORT INVIDUAL FILTER or CONVOLUTIONS:
            #b = np.squeeze(filter_bank.flt_raw[i, :, :])  # Equivalent to MATLAB's squeeze
            #np.savetxt(f'py_fltorconv_{i}.csv', b, delimiter=',')
            #image = Image.fromarray(b)
            #image.convert("RGB").save(f'py_fltorconv_{i}.png')
        self.resp_bin = np.argmax(self.image_flt, axis=0)
        self.resp_val = np.max(self.image_flt, axis=0)
        #print('---------------------')
        #print(np.mean(np.mean(self.resp_val)))
        #print(np.max(self.resp_val,axis=tuple([0,1])))

    def image_size(self):
        return self.image_raw.shape

    def show(self, n=0):
        plt.figure("image_raw"), plt.title(self.image_name), plt.imshow(self.image_raw, cmap='gray', interpolation='nearest')
        plt.figure("resp_val"), plt.imshow(self.resp_val, cmap='gray', interpolation='nearest')
        plt.figure("resp_bin"), plt.imshow(self.resp_bin, cmap='gray', interpolation='nearest')
        plt.figure("edges overlay"), plt.imshow(hsv_to_rgb(
                                                    np.dstack((self.resp_bin/float(GABOR_BINS),
                                                    self.resp_val/np.max(self.resp_val),
                                                    self.image_raw/255.0))),
                                                cmap='gray', interpolation='nearest')
        plt.show()

#-------------------------------------------------------------------------------

def do_counting(filter_img, filename):
    """creates histogram (distance, relative orientation in image, relative gradient)"""

    h, w = filter_img.image_size()
    # cutoff minor filter responses
    normalize_fac = float(filter_img.resp_val.shape[0]*filter_img.resp_val.shape[1])
    complex_before = np.sum(filter_img.resp_val)/normalize_fac

    cutoff = np.sort(filter_img.resp_val.flatten())[-10000] # get 10000th highest response for cutting of beneath

    filter_img.resp_val[filter_img.resp_val<cutoff] = 0
    ey, ex = filter_img.resp_val.nonzero()

    # visualize remaining responses
    a = np.zeros_like(filter_img.resp_val)
    a[ey,ex] = filter_img.resp_val[ey,ex]
    np.savetxt(f'csvs/py_{filename}.csv', a, delimiter=',') #SAVES PIXEL VALUES AS CSV
    plt.figure(1), plt.title('edges_'+filename), plt.imshow(a, cmap='gray', interpolation='nearest')
    plt.savefig(os.path.join(PLOT_DIR,'edges_'+filename), bbox_inches='tight')
    plt.close()

    # lookup tables to speed up calculations
    edge_dims = filter_img.resp_val.shape
    xx, yy = np.meshgrid(np.linspace(-edge_dims[1],edge_dims[1],2*edge_dims[1]+1), np.linspace(-edge_dims[0],edge_dims[0],2*edge_dims[0]+1))
    dist = np.sqrt(xx**2+yy**2)

    orientations = filter_img.resp_bin[ey,ex] #prodces 1d array of (10000,) elements
    counts = np.zeros([MAX_DIAGONAL, CIRC_BINS, GABOR_BINS])

    print("Counting", filter_img.image_name, filter_img.image_size(), "comparing", ex.size)

    for cp in range(ex.size):
        orientations_rel = orientations - orientations[cp]
        orientations_rel = np.mod(orientations_rel + GABOR_BINS, GABOR_BINS)

        distance_rel = np.round(dist[(ey-ey[cp])+edge_dims[0], (ex-ex[cp])+edge_dims[1]]).astype("uint32")

        distance_rel[distance_rel>=MAX_DIAGONAL] = MAX_DIAGONAL-1

        direction = np.round(np.arctan2(ey-ey[cp], ex-ex[cp]) / (2.0*np.pi)*CIRC_BINS + (orientations[cp]/float(GABOR_BINS)*CIRC_BINS)).astype("uint32")
        direction = np.mod(direction+CIRC_BINS, CIRC_BINS)

        #add from counts at index to filter_img 
        a = filter_img.resp_val[ey,ex] * filter_img.resp_val[ey[cp],ex[cp]]
        np.add.at(counts, tuple([distance_rel, direction, orientations_rel]), filter_img.resp_val[ey,ex] * filter_img.resp_val[ey[cp],ex[cp]])

    return counts, complex_before


def entropy(a):
    if np.sum(a)!=1.0 and np.sum(a)>0:
        a = a / np.sum(a)
    v = a>0.0
    return -np.sum(a[v] * np.log2(a[v]))


def do_statistics(counts):
    """normalize counts, do statistics"""

    # normalize by sum
    counts_sum = np.sum(counts, axis=2) + 0.00001
    normalized_counts = counts / (counts_sum[:,:,np.newaxis])
    x = normalized_counts * np.cos(BINS_VEC)
    y = normalized_counts * np.sin(BINS_VEC)
    mean_vector = np.mean(x+1j*y, axis=2)
    circular_mean_angle = np.mod(np.angle(mean_vector) + 2*np.pi, 2*np.pi)
    circular_mean_length = np.absolute(mean_vector)

    # correction as proposed by Zar 1999
    d = 2*np.pi/GABOR_BINS
    c = d / 2.0 / np.sin(d/2.0)
    circular_mean_length *= c

    d,a,_ = normalized_counts.shape
    shannon = np.zeros((d,a))
    shannon_nan = np.zeros((d,a))
    for di in range(d):
      for ai in range(a):
        shannon[di,ai] = entropy(normalized_counts[di,ai,:])
        if counts_sum[di,ai]>1:
            shannon_nan[di,ai] = shannon[di,ai]
        else:
            shannon_nan[di,ai] = np.nan

    return normalized_counts, circular_mean_angle, circular_mean_length, shannon, shannon_nan


def do_first_order(img):
    first_order_bin = np.zeros(GABOR_BINS)
    for b in range(GABOR_BINS):
        first_order_bin[b] = np.sum(img.resp_val[img.resp_bin==b])
    first_order = entropy(first_order_bin)
    return first_order


def visualize_hist_circ(normalized_counts, circular_mean_angle, circular_mean_length, shannon, filename):
    max_distance, circ_bins = MAX_DIAGONAL, normalized_counts.shape[1]

    out_hue = np.zeros([max_distance*2, max_distance*2])
    out_sat = np.ones([max_distance*2, max_distance*2])
    out_len = np.zeros([max_distance*2, max_distance*2])

    out_shannon = np.zeros([max_distance*2, max_distance*2])

    for h in range(out_hue.shape[0]):
        for w in range(out_hue.shape[1]):
            th = h-max_distance
            tw = w-max_distance
            d = int(np.sqrt(tw**2+th**2))

            if d<max_distance:
                a = int(np.mod(np.arctan2(th,tw) / (2.0*np.pi)*(circ_bins) + circ_bins, circ_bins))

                out_hue[h,w] = circular_mean_angle[d,a]
                out_len[h,w] = circular_mean_length[d,a]

                out_shannon[h,w] = shannon[d,a]
            else:
                out_hue[h,w] = 0
                out_len[h,w] = 0
                out_sat[h,w] = 0

                out_shannon[h,w] = np.nan

    hue = out_hue/(2*np.pi)
    #print "MINMAX",np.min(out_len), np.max(out_len)

    val = np.clip(out_len / (1./24.), 0.0, 1.0)
    RGB = hsv_to_rgb(np.dstack((hue, out_sat, val)))
    plt.figure(1), plt.title('circ-angle-rgb_'+filename), plt.imshow(RGB, interpolation='nearest')
    plt.savefig(os.path.join(EXPORT_DIR,'circ-angle-rgb_'+filename), bbox_inches='tight')
    crop_zoom=100
    plt.figure(2), plt.title('circ-angle-rgbzoom_'+filename), plt.imshow(RGB[max_distance-crop_zoom:max_distance+crop_zoom,max_distance-crop_zoom:max_distance+crop_zoom], interpolation='nearest')
    plt.savefig(os.path.join(EXPORT_DIR,'circ-angle-rgbzoom_'+filename), bbox_inches='tight')

    val = np.clip(out_len / (1./24.)*4, 0.0, 1.0)
    RGB = hsv_to_rgb(np.dstack((hue, out_sat, val)))
    plt.figure(3), plt.title('circ-angle-rgb-e4_'+filename), plt.imshow(RGB, interpolation='nearest')
    plt.savefig(os.path.join(EXPORT_DIR,'circ-angle-rgb-e4_'+filename), bbox_inches='tight')
    crop_zoom=100
    plt.figure(4), plt.title('circ-angle-rgbzoom-e4_'+filename), plt.imshow(RGB[max_distance-crop_zoom:max_distance+crop_zoom,max_distance-crop_zoom:max_distance+crop_zoom], interpolation='nearest')
    plt.savefig(os.path.join(EXPORT_DIR,'circ-angle-rgbzoom-e4_'+filename), bbox_inches='tight')

    plt.figure(5), plt.title('circ-angle-hue_'+filename), plt.imshow(hsv_to_rgb(np.dstack((hue, out_sat, np.ones_like(hue)))), cmap='jet', interpolation='nearest')
    plt.savefig(os.path.join(EXPORT_DIR,'circ-angle-hue_'+filename), bbox_inches='tight')
    plt.figure(6), plt.title('circ-angle-len_'+filename), plt.imshow(out_len, cmap='jet', interpolation='nearest'), plt.colorbar()
    plt.savefig(os.path.join(EXPORT_DIR,'circ-angle-len_'+filename), bbox_inches='tight')

    plt.figure(7), plt.title('circ-shannon_'+filename), plt.imshow(out_shannon, cmap='jet', interpolation='nearest'), plt.colorbar()
    plt.savefig(os.path.join(EXPORT_DIR,'circ-shannon_'+filename), bbox_inches='tight')
    plt.close("all")


def visualize_single_hist(l, filename, normalize=True):
    if normalize:
        l = l/np.max(l)

    m, minl, maxl = 250, 100, 150
    l = l + minl/float(maxl)
    gb = float(GABOR_BINS)

    img = np.ones((500, 500, 3), dtype=np.uint8)*255
    rr, cc = circle(m, m, minl+maxl)
    img[rr, cc, :] = (0.8*255, 0.8*255, 0.8*255)
    rr, cc = circle(m, m, minl)
    img[rr, cc, :] = (1.0*255, 1.0*255, 1.0*255)

    for a in range(GABOR_BINS):

        sx = np.array([m+minl*np.cos((a-1/gb*4*np.pi)/gb*2*np.pi), m+minl*np.cos((a+1/gb*4*np.pi)/gb*2*np.pi)])
        sy = np.array([m+minl*np.sin((a-1/gb*4*np.pi)/gb*2*np.pi), m+minl*np.sin((a+1/gb*4*np.pi)/gb*2*np.pi)])

        x = np.array([m+maxl*l[a]*np.cos((a+1/gb*4*np.pi)/gb*2*np.pi), m+maxl*l[a]*np.cos((a-1/gb*4*np.pi)/gb*2*np.pi)])
        y = np.array([m+maxl*l[a]*np.sin((a+1/gb*4*np.pi)/gb*2*np.pi), m+maxl*l[a]*np.sin((a-1/gb*4*np.pi)/gb*2*np.pi)])
        rr, cc = polygon(np.hstack((sx,x)), np.hstack((sy,y)))

        img[rr, cc, :] = tuple(i * 255 for i in hsv_to_rgb((a/gb,1,1)))

    plt.figure(1), plt.title('single_hist-'+filename), plt.imshow(img)
    plt.savefig(os.path.join(PLOT_DIR,'single_hist-'+filename), bbox_inches='tight')
    plt.close("all")

#-------------------------------------------------------------------------------

def get_file_list(directory, extensions, max_files=0):
    file_list = []
    for f in os.listdir(directory):
        name, file_ext = os.path.splitext(f)
        if file_ext in extensions:
            file_list.append(os.path.join(directory, name + file_ext))

    file_list = sorted(file_list)
    return file_list if max_files==0 else file_list[:max_files]


# if __name__=="__main__": 
TIMESTAMP = time.strftime("%Y.%m.%d-%H%M%S-%a")

print("Cocircus - NaN")
print(" started", TIMESTAMP)

print(" GABOR_BINS =", GABOR_BINS)
print(" CIRC_BINS = ", CIRC_BINS)
print(" MAX_PIXELS =", MAX_PIXELS)
print(" MAX_TASKS = ", MAX_TASKS, "no forking" if MAX_TASKS==0 else "forking every", MAX_TASKS, "tasks")
print(" writing to", IMAGE_DIR, PLOT_DIR, TEXT_DIR)

np.set_printoptions(precision=4, suppress=True, linewidth=160)
#     importlib.reload(sys), sys.setdefaultencoding('utf8')

filter_bank = Filter_bank(GABOR_BINS, flt_size=FILTER_SIZE)
file_list = get_file_list(IMAGE_DIR, [".tif",".jpg", ".png"])

for i in range(filter_bank.num_filters):
    #print(np.mean(filter_bank.create_gabor(FILTER_SIZE, theta=BINS_VEC[i], octave=3),tuple([0,1])))
    filter_bank.set_flt(i, filter_bank.create_gabor(FILTER_SIZE, theta=BINS_VEC[i], octave=3))
    # filter_bank.show(i)
#print('ABC')
#print(filter_bank.flt_raw[1])
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
if not os.path.exists(TEXT_DIR):
    os.makedirs(TEXT_DIR)
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

log_file = os.path.join(EXPORT_DIR,"export-"+TIMESTAMP+".csv")

RANGES = [(20,80), (80,160), (160,240)]
result_names = ["%s%d-%d" % (n, r[0], r[1]) for r in RANGES for n in ["avg-shannon","avg-shannon-nan"]]
result_names.append("first_order-shannon")
result_names.append("edge-density")

with open(log_file, 'w') as log:
    log.write("image,"+",".join(result_names)+"\n")

i, is_child = 0, False
while i < len(file_list):
    if is_child or MAX_TASKS==0:
        print(" (%d/%s)" % (i+1, len(file_list)), end=' ')
        img = FilterImage(file_list[i], max_pixels=MAX_PIXELS)
        img.run_filterbank(filter_bank)
        counts, complex_before = do_counting(img, os.path.basename(file_list[i]))

        normalized_counts, circular_mean_angle, circular_mean_length, shannon, shannon_nan = do_statistics(counts)
        #visualize_hist_circ(normalized_counts, circular_mean_angle, circular_mean_length, shannon, os.path.basename(file_list[i]))

        results = []
        for r in RANGES:
            results.append(np.mean(np.mean(shannon, axis=1)[r[0]:r[1]]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                results.append(np.nanmean(np.nanmean(shannon_nan, axis=1)[r[0]:r[1]]))
        results.append(do_first_order(img))
        results.append(complex_before)

        np.savetxt(os.path.join(TEXT_DIR, os.path.basename(file_list[i])+".shannon.txt"), np.mean(shannon, axis=1))
        np.savetxt(os.path.join(TEXT_DIR, os.path.basename(file_list[i])+".shannon-nan.txt"), np.nanmean(shannon_nan, axis=1))

        with open(log_file, 'a') as log:
            log.write(os.path.basename(file_list[i])+","+",".join([str(v) for v in results])+"\n")
        i += 1
        if MAX_TASKS != 0 and (i % MAX_TASKS==0 or i==len(file_list)):
            print("child: I'm leaving")
            sys.exit(0)
    else:
        child_pid = os.fork()
        if child_pid==0:
            print("child: I'm doing work, my pid is",child_pid)
            is_child = True
        else:
            pid, status = os.waitpid(child_pid, 0)
            print("parent: wait returned, pid = %d, status = %d" % (pid, status))
            if status != 0:
                print("parent: child terminated with an error, parent quitting.")
                sys.exit(1)
            i += MAX_TASKS
