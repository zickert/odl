import os
from odl.contrib.mrc import FileReaderMRC
import odl
import numpy as np


gamma_huber = 0.1
reg_par = 1e-5
iterate = 0


temp_path = '/home/zickert/TEM_reco_project/Reconstructions/Simulated/Balls/dose_6000/gradient_descent_huber_reg'



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
