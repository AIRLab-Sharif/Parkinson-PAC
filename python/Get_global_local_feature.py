import pandas as pd
import numpy as np
import scipy.signal as ss

def conv_PAC_dist(PAC_dist,h=5,w=5,down_sample_factor_h=5,down_sample_factor_w=5,filter_mode='valid'):
    """PAC_dist input PAC values\
    h filter kernel size along axis 0\
    w filter kernel size along axis 1\
    down sample factor h along axis 0\
    down sample factor w along axis 1\
    filter mode define the mode of convolve2d of scipy.signal"""
    # Filtering
    width_filter = w
    height_filter = h
    kernel_filter = np.ones(shape=[width_filter,height_filter])/(width_filter*height_filter)
    PAC_feature = ss.convolve2d(PAC_dist,kernel_filter,mode=filter_mode)

    # Sampling
    H = PAC_dist.shape[0]//down_sample_factor_h
    Num_H_samples = (H-1)*down_sample_factor_h
    W = PAC_dist.shape[1]//down_sample_factor_w
    Num_W_samples = (W-1)*down_sample_factor_w
    height_index = np.linspace(0,Num_H_samples,H).astype(np.int32)
    width_index = np.linspace(0,Num_W_samples,W).astype(np.int32)
    w,h = np.meshgrid(width_index,height_index)

    return PAC_feature[h,w]
