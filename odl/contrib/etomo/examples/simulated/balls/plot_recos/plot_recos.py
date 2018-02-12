import os
from odl.contrib.mrc import FileReaderMRC
import odl
import numpy as np


gamma_huber = 0.01
reg_par = 1e-3
iterate = 800


temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg'
#temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg_LINEARIZED'



path = temp_path +'_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_' + str(iterate) + '.npy'



rescale_factor = 1e9

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*210e-9/4,
                                       -rescale_factor*250e-9/4,
                                       -rescale_factor*40e-9/4],
                               max_pt=[rescale_factor*210e-9/4,
                                       rescale_factor*250e-9/4,
                                       rescale_factor*40e-9/4],
                               shape=[210, 250, 40], dtype='float64')

reco_array = np.load(path)
reco = reco_space.element(reco_array)
reco.show()


# %%

gamma_huber = 0.01
reg_par = 1e-3
iterate = 800


#temp_path2 = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg'
temp_path2 = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg_LINEARIZED'



path2 = temp_path2 +'_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_' + str(iterate) + '.npy'



rescale_factor = 1e9

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*210e-9/4,
                                       -rescale_factor*250e-9/4,
                                       -rescale_factor*40e-9/4],
                               max_pt=[rescale_factor*210e-9/4,
                                       rescale_factor*250e-9/4,
                                       rescale_factor*40e-9/4],
                               shape=[210, 250, 40], dtype='float64')

reco_array2 = np.load(path2)
reco2 = reco_space.element(reco_array2)
reco2.show()

diff = reco-reco2
diff.show()


# %% 

#gamma_huber = 0.01
#reg_par = 5e-4
iterate = 1000


step_param = 1e-4
reg_param = 0.1

temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/pdhg_tv_pos_constr_LINEARIZED'


#temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg'
#temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg_LINEARIZED'



path = temp_path +'_step_par='+str(step_param)+'_reg_par='+str(reg_param)+'/iterate_' + str(iterate) + '.npy'



rescale_factor = 1e9

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*210e-9/4,
                                       -rescale_factor*250e-9/4,
                                       -rescale_factor*40e-9/4],
                               max_pt=[rescale_factor*210e-9/4,
                                       rescale_factor*250e-9/4,
                                       rescale_factor*40e-9/4],
                               shape=[210, 250, 40], dtype='float64')

reco_array = np.load(path)
reco = reco_space.element(reco_array)
reco.show()


# %% 

#gamma_huber = 0.01
#reg_par = 5e-4
iterate = 1000


step_param = 1e-4
reg_param = 0.1

temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/pdhg_tv_pos_constr'


#temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg'
#temp_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg_LINEARIZED'



path = temp_path +'_step_par='+str(step_param)+'_reg_par='+str(reg_param)+'/iterate_' + str(iterate) + '.npy'



rescale_factor = 1e9

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*210e-9/4,
                                       -rescale_factor*250e-9/4,
                                       -rescale_factor*40e-9/4],
                               max_pt=[rescale_factor*210e-9/4,
                                       rescale_factor*250e-9/4,
                                       rescale_factor*40e-9/4],
                               shape=[210, 250, 40], dtype='float64')

reco_array = np.load(path)
reco = reco_space.element(reco_array)
reco.show()