import odl
import numpy as np
from odl.contrib.mrc import FileReaderMRC
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


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
dir_path = os.path.abspath('/mnt/imagingnas/data/Users/gzickert/TEM/Data/Simulated/Balls/dose_6000')
file_path_phantom = os.path.join(dir_path, 'balls_phantom.mrc')
with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()         
phantom = reco_space.element(phantom_asarray)
# We remove the background from the phantom
bg_cst = np.min(phantom)
phantom -= bg_cst

#%%

iterate = 182
num_cycles = 3

reg_param_list = [1e-3, 3e2, 9e2, 3e3, 9e3, 3e4]
gamma_H1_list = [0.0, 0.9, 0.95, 0.99]
Niter_CG_list = [20, 30, 40]

for reg_param in reg_param_list:
    for gamma_H1 in gamma_H1_list:
        for Niter_CG in Niter_CG_list:
            try:
                method_path = 'Simulated/Balls/no_noise/kaczmarz'
                param_path = '/gamma_H1='+str(gamma_H1)+'_reg_par='+str(reg_param)+'_niter_CG='+str(Niter_CG)+'_num_cycles='+str(num_cycles)+'_iterate_' + str(iterate) 
                path = base_path + method_path + param_path + '.npy'
                fig_path = base_path + method_path + param_path + '.png'
               
                reco_array = np.load(path)
                reco = reco_space.element(reco_array)
                reco.show(title=method_path+'\n'+param_path+'\n'
                          +'SSIM='+str(ssim(phantom.asarray(), reco.asarray()))
                          +', PSNR='+str(psnr(phantom.asarray(), reco.asarray(),
                                                   dynamic_range=np.max(phantom) - np.min(phantom))),
                                              saveto=fig_path)
            except:
                pass

# %%
iterate = 1000

step_param_list = [1e-4, 1e-3, 1e-2, 1e-1]
reg_param_list = [1e-3]

for step_param in step_param_list:
    for reg_param in reg_param_list:
      
        method_path = 'Simulated/Balls/dose_6000/pdhg_tv_pos_constr'
        param_path = '_step_par='+str(step_param)+'_reg_par='+str(reg_param)+'/iterate_' + str(iterate) 
        path = base_path + method_path + param_path + '.npy'
        fig_path = base_path + method_path + param_path
        
        reco_array = np.load(path)
        reco = reco_space.element(reco_array)
        reco.show(title=method_path+'\n'+param_path+'\n'
                  +'SSIM='+str(ssim(phantom.asarray(), reco.asarray()))
                  +', PSNR='+str(psnr(phantom.asarray(), reco.asarray(),
                                           dynamic_range=np.max(phantom) - np.min(phantom)))
                  )#, saveto=fig_path)


# %%
iterate = 1000

step_param_list = [1e-4, 1e-3, 1e-2, 1e-1]
reg_param_list = [1e-3]

for step_param in step_param_list:
    for reg_param in reg_param_list:
      
        method_path = 'Simulated/Balls/dose_6000/pdhg_tv_pos_constr_LINEARIZED'
        param_path = '_step_par='+str(step_param)+'_reg_par='+str(reg_param)+'/iterate_' + str(iterate) 
        path = base_path + method_path + param_path + '.npy'
        fig_path = base_path + method_path + param_path
        
        reco_array = np.load(path)
        reco = reco_space.element(reco_array)
        reco.show(title=method_path+'\n'+param_path+'\n'
                  +'SSIM='+str(ssim(phantom.asarray(), reco.asarray()))
                  +', PSNR='+str(psnr(phantom.asarray(), reco.asarray(),
                                           dynamic_range=np.max(phantom) - np.min(phantom)))
                  )#, saveto=fig_path)


#%%
reg_par_list = [5e-4, 2.5e-4, 7.5e-4]
gamma_huber_list = [1e-2]
iterate = 1000





for gamma_huber in gamma_huber_list:
    for reg_par in reg_par_list:
      
        method_path = '/Simulated/Balls/dose_6000/gradient_descent_huber_reg'
        param_path = '_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_' + str(iterate) 
        path = base_path + method_path + param_path + '.npy'
        fig_path = base_path + method_path + param_path
        
        reco_array = np.load(path)
        reco = reco_space.element(reco_array)
        reco.show(title=method_path+'\n'+param_path+'\n'
                  +'SSIM='+str(ssim(phantom.asarray(), reco.asarray()))
                  +', PSNR='+str(psnr(phantom.asarray(), reco.asarray(),
                                           dynamic_range=np.max(phantom) - np.min(phantom)))
                  )#, saveto=fig_path)



#%%
reg_par_list = [1e-4, 1e-3, 1e-2, 1e-1, 5e-3, 5e-4, 2.5e-4, 7.5e-4]
gamma_huber_list = [1e-2]
iterate = 800


for gamma_huber in gamma_huber_list:
    for reg_par in reg_par_list:
        try:
            method_path = '/Simulated/Balls/dose_6000/gradient_descent_huber_reg_LINEARIZED'
            param_path = '_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_' + str(iterate) 
            path = base_path + method_path + param_path + '.npy'
            fig_path = base_path + method_path + param_path
            
            reco_array = np.load(path)
            reco = reco_space.element(reco_array)
            reco.show(title=method_path+'\n'+param_path+'\n'
                      +'SSIM='+str(ssim(phantom.asarray(), reco.asarray()))
                      +', PSNR='+str(psnr(phantom.asarray(), reco.asarray(),
                                               dynamic_range=np.max(phantom) - np.min(phantom)))
                      )#, saveto=fig_path)
        except:
            pass

