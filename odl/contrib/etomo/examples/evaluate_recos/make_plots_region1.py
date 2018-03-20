import odl
import numpy as np
from odl.contrib import fom
from odl.contrib.mrc import FileReaderMRC


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
num_cycles = 3
num_angles = 81
iterate = (num_cycles * num_angles) - 1


reg_param_list = [3e2, 9e2, 3e3, 9e3, 3e4]
gamma_H1_list = [0.8, 0.9, 0.95, 0.99]
Niter_CG_list = [30]

for reg_param in reg_param_list:
    for gamma_H1 in gamma_H1_list:
        for Niter_CG in Niter_CG_list:
            try:
                method_path = 'Experimental/Region1/kaczmarz'
                param_path = '/gamma_H1='+str(gamma_H1)+'_reg_par='+str(reg_param)+'_niter_CG='+str(Niter_CG)+'_num_cycles='+str(num_cycles)+'_iterate_' + str(iterate) 
                path = base_path + method_path + param_path + '.npy'
                fig_path = base_path + method_path + param_path + '.png'

                reco_array = np.load(path) 
                reco = reco_space.element(reco_array)
                reco.show(title=method_path+'\n'+param_path, saveto=fig_path)
            except:
                pass



#%%
                
            
iterate = 1000

#step_param_list = [1e-4, 1e-3, 1e-2, 1e-1]
#reg_param_list = [1e-4]



reg_param_list = [1e-4, 3e-4, 1e-3, 3e-3]
step_param_list = [1e-3, 1e-2, 1e-1]




for step_param in step_param_list:
    for reg_param in reg_param_list:
        try:
            method_path = 'Experimental/Region1/pdhg_tv_pos_constr'
            param_path = '/step_par='+str(step_param)+'_reg_par='+str(reg_param)+'_iterate_' + str(iterate) 
            path = base_path + method_path + param_path + '.npy'
            fig_path = base_path + method_path + param_path + '.png'
            
            reco_array = np.load(path)
            reco = reco_space.element(reco_array)
            reco.show(title=method_path+'\n'+param_path
                      , saveto=fig_path)
        except:
            pass



#%%
#reg_par_list = [5e-4, 2.5e-4, 7.5e-4]
#gamma_huber_list = [1e-2]


reg_par_list = [2.5e-4, 3.75e-4, 5e-4, 6.25e-4, 7.5e-4]
gamma_huber_list = [1e-2]
iterate_list = [1000]

method_path = 'Experimental/Region1/gradient_descent_huber_reg'

for iterate in iterate_list:
    for gamma_huber in gamma_huber_list:
        for reg_par in reg_par_list:
          
            param_path = '/_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'_iterate_' + str(iterate) 
            path = base_path + method_path + param_path + '.npy'
            fig_path = base_path + method_path + param_path + '.png'
            
            reco_array = np.load(path)
            reco = reco_space.element(reco_array)
            reco.show(title=method_path+'\n'+param_path,
                      saveto=fig_path)
    
    

