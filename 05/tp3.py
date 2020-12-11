import numpy as np

def convolute_image(im, kernel, operation):
    h,w = im.shape
    
    kernel_h, kernel_w = kernel.shape
    
    even_kernel = kernel_h%2 == 0
    kernel_h_off, kernel_w_off = kernel_h//2,  kernel_w//2
    
    resize_im = np.pad(im, (kernel_h_off,kernel_w_off), mode='edge')
    
    im_out = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            if even_kernel:
                i_off = i+kernel_h_off+1
                j_off = j+kernel_w_off+1
            else:
                i_off = i+2*kernel_h_off+1
                j_off = j+2*kernel_w_off+1
            portion = resize_im[
                i: i_off,
                j: j_off
            ]
            im_out[i,j] = operation(portion, kernel)
            
    return im_out

def kernel_sum(portion, kernel):
    return np.sum(kernel*portion)
    
def low_pass(im, order=1):
    kernel = 1/9 * np.ones((2*order+1, 2*order+1))
    
    return convolute_image(im,kernel,kernel_sum)

def high_pass(im, alpha=1, order=1):
    kernel = 1/9 * np.array([
        [-1,-1,-1],
        [-1, 8,-1],
        [-1,-1,-1]
    ])

    return (im+convolute_image(im, kernel, kernel_sum)).clip(0,255)

def sharpening(im):
    kernel = np.array([
        [0   ,-1/4,    0],
        [-1/4,   2, -1/4],
        [0   ,-1/4,    0],
    ])
    
    return convolute_image(im, kernel, kernel_sum).clip(0,255)

def median_filter(im, order=1):
    kernel = np.ones((2*order+1, 2*order+1))
    def median (portion, kernel):
        return np.median(portion)
    return convolute_image(im, kernel, median)

def roberts(im, g):
    kernel = np.array([
        [1, 0],
        [0,-1]
    ])
    if g=='y':
        kernel = np.rot90(kernel)
    elif g!='x':
        raise ValueError('The parameter g hast to be "x" or "y"')
    
    return np.clip(convolute_image(im, kernel, kernel_sum),0,255)

def prewitt(im, g):
    kernel = np.array([
        [-1,-1,-1],
        [ 0, 0, 0],
        [ 1, 1, 1]
    ])
    
    if g=='y':
        kernel = np.transpose(kernel)
    elif g!='x':
        raise ValueError('The parameter g hast to be "x" or "y"')
    
    return np.clip(convolute_image(im, kernel, kernel_sum),0,255)

def sobel(im, g):
    kernel = np.array([
        [-1,-2,-1],
        [ 0, 0, 0],
        [ 1, 2, 1]
    ])
    
    if g=='y':
        kernel = np.transpose(kernel)
    elif g!='x':
        raise ValueError('The parameter g hast to be "x" or "y"')
    
    return np.clip(convolute_image(im, kernel, kernel_sum),0,255)

def laplace(im):
    kernel = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ])
    return convolute_image(im, kernel, kernel_sum)

def fft_image(im):
    unshift_fft = np.fft.fft2(im)
    fft = np.fft.fftshift(unshift_fft)
    
    return fft

def ifft_image(fft):
    unshift = np.fft.ifftshift(fft)
    return np.abs(np.fft.ifft2(unshift))

def distance(p1, p2):
    x1,y1 = p1[:2]
    x2,y2 = p2[:2]
    
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def low_pass_ideal(p, center, D0):
    if distance(p,center)<=D0:
        return 1
    else:
        return 0
    
def low_pass_butterworth(p, center, D0, **kwargs):
    n = kwargs['n']
    return 1/(1+(distance(p,center)/D0)**(2*n))

def low_pass_gaussian(p, center, D0):
    return np.exp(((-distance(p,center)**2)/(2*(D0**2))))

low_pass_f ={
    'ideal': low_pass_ideal,
    'butterworth':low_pass_butterworth,
    'gaussian':low_pass_gaussian
}

def high_pass_ideal(p, center, D0):
    if distance(p,center)<=D0:
        return 0
    else:
        return 1
    
def high_pass_butterworth(p, center, D0, **kwargs):
    n = kwargs['n']
    return 1-1/(1+(distance(p,center)/D0)**(2*n))

def high_pass_gaussian(p, center, D0):
    return 1 - np.exp(((-distance(p,center)**2)/(2*(D0**2))))

high_pass_f ={
    'ideal': high_pass_ideal,
    'butterworth':high_pass_butterworth,
    'gaussian':high_pass_gaussian
}

def freq_mask(D0, im_shape,fs,shape='ideal', **kwargs):
    rows, cols =im_shape[:2]
    base = np.ones((rows,cols))
    center = (rows/2, cols/2)
    
    for i in range(rows):
        for j in range(cols):
            base[i,j] = fs[shape]((i,j), center, D0, **kwargs)
            
    return base

def low_pass_mask(D0, im_shape, shape='ideal', **kwargs):
    return freq_mask(D0, im_shape, low_pass_f, shape, **kwargs)


def high_pass_mask(D0, im_shape, shape='ideal', **kwargs):
    return freq_mask(D0, im_shape, high_pass_f, shape, **kwargs)

def apply_freq_filter(im, mask):
    fft_im = fft_image(im)

    filtered = fft_im*mask
    
    return ifft_image(filtered)

def lp_ideal(im, D0=100):
    return apply_freq_filter(im, low_pass_mask(D0, im.shape, shape='ideal'))
def lp_butterworth(im, D0=100,n=2):
    return apply_freq_filter(im, low_pass_mask(D0, im.shape, shape='butterworth',n=n))
def lp_gaussian(im, D0=100):
    return apply_freq_filter(im, low_pass_mask(D0, im.shape, shape='gaussian'))
def hp_ideal(im, D0=100):
    return apply_freq_filter(im, high_pass_mask(D0, im.shape, shape='ideal'))
def hp_butterworth(im, D0=100,n=2):
    return apply_freq_filter(im, high_pass_mask(D0, im.shape, shape='butterworth',n=n))
def hp_gaussian(im, D0=100):
    return apply_freq_filter(im, high_pass_mask(D0, im.shape, shape='gaussian'))

