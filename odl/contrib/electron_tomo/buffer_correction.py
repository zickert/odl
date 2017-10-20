# -*- coding: utf-8 -*-

import numpy as np

def buffer_correction(data, coords=[[0,0.25],[0,0.25]]):
    

    dim_t, dim_x, dim_y = data.shape
    data_asarray = data.asarray()
    background = data_asarray[:,round(dim_x*coords[0][0]):round(dim_x*coords[0][1]), 
                                round(dim_y*coords[1][0]):round(dim_y*coords[1][1])]
    print(background.shape)
    bg_mean = np.mean(background, (1,2))
    return data.space.element(data_asarray * (1.0/bg_mean.reshape(dim_t,1,1)))