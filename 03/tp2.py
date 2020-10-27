import imageio
import matplotlib.pyplot as plt
import numpy as np

def bin_im_op(im_a, im_b, operation):
    im_a_h,im_a_w = im_a.shape
    im_b_h,im_b_w = im_b.shape

    im_out_h = max(im_a_h, im_b_h)
    im_out_w = max(im_a_w, im_b_w)

    im_out = np.zeros((im_out_h, im_out_w))

    for i, row in enumerate(im_a):
        for j, _ in enumerate(row):
            im_out[i][j] = im_a[i][j]
    
    for i, row in enumerate(im_b):
        for j, _ in enumerate(row):
            im_out[i][j] = operation(im_out[i][j],im_b[i][j])
            
    return np.clip(im_out, 0, 255)

def im_add(im_a, im_b):
    return bin_im_op(im_a, im_b, lambda x,y: x+y)

def im_add(im_a, im_b):
    return bin_im_op(im_a, im_b, lambda x,y: x+y)

def scalar_op(im, scalar):
    int_im = im.astype(np.uint64)
    return np.clip(int_im*scalar, 0, 255).astype(np.uint8)

def im_avg(imgs):
    accum = np.zeros((10, 10))
    
    for i in imgs:
        accum = im_add(accum, i)
        
    return accum/len(imgs)

def bit_im_op(im_a, im_b,op):
    return bin_im_op(im_a, im_b, op)

def im_and(im_a, im_b, zero=0, one=1):
    return bit_im_op(im_a, im_b, lambda x,y: (x==one) and (y==one))

def im_or(im_a, im_b, zero=0, one=1):
    return bit_im_op(im_a, im_b,lambda x,y: (x==one) or (y==one))

def horizontal_reflection(im):
    h,w = im.shape
    im_out = np.zeros((h,w))
    
    for i, row in enumerate(im):
        for j, p in enumerate(row):
            im_out[i][j] = im[i][w-j-1]
    
    return im_out

def vertical_reflection(im):
    h,w = im.shape
    im_out = np.zeros((h,w))
    
    for i, row in enumerate(im):
        for j, p in enumerate(row):
            im_out[i][j] = im[h-i-1][j]
    
    return im_out

def double_reflection(im):
    return vertical_reflection(horizontal_reflection(im))

def rotation_90(im):
    h,w = im.shape
    im_out = np.zeros((w,h))
    
    for i, row in enumerate(im):
        for j, p in enumerate(row):
            im_out[j][h-i-1] = im[i][j]
    
    return im_out

def rotation_180(im):
    h,w = im.shape
    im_out = np.zeros((h,w))
    
    for i, row in enumerate(im):
        for j, p in enumerate(row):
            im_out[h-1-i][w-1-j] = im[i][j]
    
    return im_out

def src_dst_scaling(im, factors):
    sx, sy = factors
    
    h,w = im.shape
    newH, newW = int(h*sy), int(w*sx)
    
    im_out = np.zeros((newH, newW))

    for i, row in enumerate(im):
        for j, p in enumerate(row):
            iprime=min(int(i*sy), newH-1)
            jprime=min(int(j*sx), newW-1)
            im_out[iprime][jprime] = im[i][j]
    return im_out

def src_dst_traslation(im, tvalues):
    tx, ty = tvalues
    
    h,w = im.shape
    
    im_out = np.zeros((h,w))

    for i, row in enumerate(im):
        for j, p in enumerate(row):
            try:
                jprime = int(j+tx)
                iprime = int(i+ty)
                if iprime < 0 or jprime < 0:
                    continue
                im_out[iprime][jprime] = im[i][j]
            except IndexError:
                continue
    return im_out

def src_dst_rotation(im, angle):
    h,w = im.shape
    
    im_out = np.zeros((h,w))
    for i, row in enumerate(im):
        for j, p in enumerate(row):
            try:
                jprime = int((j-w/2)*np.cos(angle)+(i-h/2)*np.sin(angle)+(w/2))
                iprime = int(-((j-w/2)*np.sin(angle))+(i-h/2)*np.cos(angle)+(h/2))
                if iprime< 0 or jprime < 0:
                    continue
                im_out[iprime][jprime] = im[i][j]
            except IndexError:
                continue
    return im_out

def dst_src_scaling(im, factors):
    sx,sy = factors
    
    h,w = im.shape
    
    newH, newW = (int(h*sy), int(w*sx))
    
    im_out = np.zeros((newH, newW))
    
    for i,row in enumerate(im_out):
        for j, p in enumerate(row):
            src_i = int(i//sy)
            src_j = int(j//sx)
            im_out[i,j] = im[src_i,src_j]
    
    return im_out

def dst_src_traslation(im, tvalues):
    tx, ty = tvalues
    
    h,w = im.shape
    
    im_out = np.zeros((h,w))

    for i, row in enumerate(im):
        for j, p in enumerate(row):
            try:
                src_j = int(j-tx)
                src_i = int(i-ty)
                if src_i < 0 or src_j < 0:
                    continue
                im_out[i][j] = im[src_i][src_j]
            except IndexError:
                continue
    return im_out

def dst_src_rotation(im, angle):
    h,w = im.shape
    
    angle = -angle
    
    im_out = np.zeros((h,w))
    for i, row in enumerate(im):
        for j, p in enumerate(row):
            try:
                src_j = int((j-w/2)*np.cos(angle)-(i-h/2)*np.sin(angle)+(w/2))
                src_i = int(((j-w/2)*np.sin(angle))+(i-h/2)*np.cos(angle)+(h/2))
                if src_i< 0 or src_j < 0:
                    continue
                im_out[src_i][src_j] = im[i][j]
            except IndexError:
                continue
    return im_out