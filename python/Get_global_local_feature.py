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
    height_index = np.linspace(0,PAC_feature.shape[0]-1,PAC_dist.shape[0]//down_sample_factor_h).astype(np.int32)
    width_index = np.linspace(0,PAC_feature.shape[1]-1,PAC_dist.shape[1]//down_sample_factor_w).astype(np.int32)
    h,w = np.meshgrid(height_index,width_index)
    return PAC_feature[h,w]
