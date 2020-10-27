# Funciones desarrolladas en el tp1
# Importamos imageio
import imageio

# Numpy para array and matrix operations
import numpy as np

# Imprortamos matplotlib para poder visualizar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def negative(im):
    return 255 - im

def lineal(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    def f(x):
        m = (y2 - y1)/(x2-x1)
        b = y1 - (x1*m)
        return m*x + b
    return f

def lineal_by_parts(parts):
    def f(x):
        index = 0
        for i,p in enumerate(parts[:-1]):
            
            if p[0] > x:
                break
            index = i
        return lineal(parts[index], parts[index+1])(x)
    return f

def contrast_stretch(im, p1,p2):
    x1,y1 = p1
    x2,y1 = p2
    norm_im = im / 255
    out_im = norm_im.copy()
    
    parts = [(0,0), p1, p2, (1,1)]
    for i, row in enumerate(norm_im):
        for j, col in enumerate(row):
            out_im[i][j] = lineal_by_parts(parts)(norm_im[i][j])
    return out_im

def bit_plane_slice(im, bit_number):
    return im & (2**(bit_number-1))

def threshold(im, thresh=.5, norm=False):
    s = im
    if not norm:
        s = s/255
    s[s > thresh] = 1
    s[s<= thresh] =0
    return s

# Cálculo del histograma
def image_histogram(im, bins=255):
    dx = 1 / bins

    dxs = [b*(dx) for b in range(bins+1)]
    ranges = [z for z in zip(dxs[:-1], dxs[1:])]

    h,w = im.shape
    im1d = im.reshape(h*w) / 255

    hist = np.zeros((bins))
    
    
    def get_interval_check(lower, upper):
        
        def start_case(v):
            return v>=lower and v <= upper
        def non_start_case(v):
            return v>lower and v<=upper

        if lower == 0:
            return start_case
        else:
            return non_start_case


    for p in im1d:
        for i, r in enumerate(ranges):
            if get_interval_check(*r)(p):
                hist[i] += 1
    
    return hist, dxs

def plot_histogram(im, n_interval = 10):
    hist, bins = image_histogram(im, bins=n_interval)
    h,w = im.shape
    number_of_pixels = h*w
    percent = [(height/number_of_pixels)*100 for height in hist]

    cdf = hist.cumsum()
    norm_cdf = cdf/cdf.max()

    fig, ax = plt.subplots(1, 2, figsize=(15,10))
    ax[0].imshow(im, cmap='gray')
    ax2 = ax[1].twinx()
    ax2.plot(np.arange(0,1,1/n_interval),norm_cdf,color='r')
    ax[1].bar(bins[:-1], percent, width=1/n_interval,align="edge")
    plt.show()

def equalize(hist, im):
    cdf = hist.cumsum()
    norm_cdf = cdf/cdf.max()
    cdf_eq = np.ma.masked_equal(cdf,0)
    cdf_eq = (cdf_eq - cdf_eq.min())*255/(cdf_eq.max()-cdf_eq.min())
    im_eq = np.ma.filled(cdf_eq,0)[im]
    return im_eq

