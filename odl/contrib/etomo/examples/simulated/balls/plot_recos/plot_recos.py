import odl
import numpy as np

rescale_factor = 1e9

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*210e-9/4,
                                       -rescale_factor*250e-9/4,
                                       -rescale_factor*40e-9/4],
                               max_pt=[rescale_factor*210e-9/4,
                                       rescale_factor*250e-9/4,
                                       rescale_factor*40e-9/4],
                               shape=[210, 250, 40], dtype='float64')


base_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/'

# %%

gamma_huber_list = []
reg_par_list = []

for gamma_huber in gamma_huber_list:
    for reg_par in reg_par_list:
        gamma_huber = 0.01
        reg_par = 1e-3
        iterate = 800
        
        method_path = '/Simulated/Balls/dose_6000/gradient_descent_huber_reg'
        param_path = '_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_' + str(iterate) + '.npy'
        path = base_path + method_path + param_path
        
        
        
        reco_array = np.load(path)
        reco = reco_space.element(reco_array)
        reco.show(title=method_path+'\n'+param_path)


# %%
gamma_huber = 0.01
reg_par = 1e-3
iterate = 800


method_path = '/Simulated/Balls/dose_6000/gradient_descent_huber_reg_LINEARIZED'
param_path = '_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_' + str(iterate) + '.npy'
path = base_path + method_path + param_path

reco_array = np.load(path)
reco = reco_space.element(reco_array)
reco.show(title=method_path+'\n'+param_path)


# %% 
iterate = 1000

step_param = 1e-4
reg_param = 0.1


method_path = 'Simulated/Balls/dose_6000/pdhg_tv_pos_constr'
param_path = '_step_par='+str(step_param)+'_reg_par='+str(reg_param)+'/iterate_' + str(iterate) + '.npy'
path = base_path + method_path + param_path


reco_array = np.load(path)
reco = reco_space.element(reco_array)
reco.show(title=method_path+'\n'+param_path)

# %% 
iterate = 1000

step_param_list = [1e-4, 1e-3, 1e-2, 1e-1]
reg_param_list = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

for step_param in step_param_list:
    for reg_param in reg_param_list:
      
        method_path = 'Simulated/Balls/dose_6000/pdhg_tv_pos_constr_LINEARIZED'
        param_path = '_step_par='+str(step_param)+'_reg_par='+str(reg_param)+'/iterate_' + str(iterate) 
        path = base_path + method_path + param_path + '.npy'
        fig_path = base_path + method_path + param_path
        
        reco_array = np.load(path)
        reco = reco_space.element(reco_array)
        reco.show(title=method_path+'\n'+param_path, saveto=figpath)

