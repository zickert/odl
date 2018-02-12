import odl
import numpy as np


gamma_huber = 0.01
reg_par = 1e-4
iterate = 1000


temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Experimental/Region1/gradient_descent_huber_reg_LINEARIZED'
path = temp_path +'_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_' + str(iterate) + '.npy'

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
reco_array = np.load(path)
reco = reco_space.element(reco_array)
reco.show()
