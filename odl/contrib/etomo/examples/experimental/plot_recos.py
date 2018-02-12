import odl
import numpy as np


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

# %%

gamma_huber_list = [0.01]
reg_par_list = [0.0001, 0.0005]
iterate = 1000

for gamma_huber in gamma_huber_list:
    for reg_par in reg_par_list:

        method_path = 'Experimental/Region1/gradient_descent_huber_reg_LINEARIZED'
        param_path = '_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_' + str(iterate) 
        path = base_path + method_path + param_path + '.npy'
        fig_path = base_path + method_path + param_path

        reco_array = np.load(path)
        reco = reco_space.element(reco_array)
        reco.show(title=method_path+'\n'+param_path, saveto=fig_path)
