import os
from odl.contrib.mrc import FileReaderMRC
import odl
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
from skimage import exposure
dir_path = os.path.abspath('/home/z/i/zickert/Recos_math_of_ET/Region1/SIRT')

file_path_reco = os.path.join(dir_path, 'FEI_region1_IMOD_SIRT_rec_reg010_cutoff005_iter10.mrc')


with mrcfile.open(file_path_reco) as mrc:
    reco = mrc.data


reco = np.transpose(reco, axes=(2, 1, 0))

#with FileReaderMRC(file_path_reco) as reco_reader:
#    reco_data = reco_reader.read_data(swap_axes=False)
#    reco_header = reco_reader.read_header()


#
voxel_size = 0.4767e-9  # m

rescale_factor = 1e9

nx = 512
ny = 256 # y is tilt-axis
nz = 200 # z is optical axis

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*(nx/2)*voxel_size,
                                       -rescale_factor*(ny/2)*voxel_size,
                                       -rescale_factor*(nz/2)*voxel_size],
                               max_pt=[rescale_factor*(nx/2)*voxel_size,
                                       rescale_factor*(ny/2)*voxel_size,
                                       rescale_factor*(nz/2)*voxel_size],
                               shape=[nx, ny, nz], dtype='float64')
    
plt.axis('off')


orthoslice = reco[:,:,(nz//2) -1].T

orthoslice = exposure.equalize_hist(orthoslice)

plt.imshow(orthoslice,origin='lower',cmap='bone')
reco = reco_space.element(reco)
reco.show()
