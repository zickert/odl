import odl
import numpy as np
from odl.contrib.mrc import FileReaderMRC
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


rescale_factor = 1e9

voxel_size = 0.4767e-9  # m

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*256*voxel_size,
                                       -rescale_factor*128*voxel_size,
                                       -rescale_factor*256*voxel_size],
                               max_pt=[rescale_factor*256*voxel_size,
                                       rescale_factor*128*voxel_size,
                                       rescale_factor*256*voxel_size],
                               shape=[512, 256, 512], dtype='float64')


base_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/'

#%%
# Implement FCS?