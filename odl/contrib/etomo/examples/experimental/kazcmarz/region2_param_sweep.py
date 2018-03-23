"""Electron tomography reconstruction example using real data."""

import numpy as np
import os
import odl
from odl.contrib import etomo
import matplotlib.pyplot as plt 
from odl.contrib.mrc import FileReaderMRC
from time import time
from datetime import timedelta
# Read data
dir_path = os.path.abspath('/mnt/imagingnas/data/Users/gzickert/TEM/Data/Experimental')
file_path_data = os.path.join(dir_path, 'region2.mrc')
angle_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Data/Experimental/tiltangles.txt'

# load tiltangles
angles = np.loadtxt(angle_path, skiprows=3, unpack=True)
angles = angles[1,:]
with FileReaderMRC(file_path_data) as reader:
    header, data = reader.read()

# The reconstruction space will be rescaled according to rescale_factor in
# order to avoid numerical issues related to having a very small reco space.
rescale_factor = 1e9

#  Define some physical constants
e_mass = 9.11e-31  # kg
e_charge = 1.602e-19  # C
planck_bar = 1.059571e-34  # Js/rad
wave_length = 0.00196e-9  # m
wave_number = 2 * np.pi / wave_length
sigma = e_mass * e_charge / (wave_number * planck_bar ** 2)

abs_phase_ratio = 0.1
obj_magnitude = sigma / rescale_factor
regpar = 3e3
num_angles = 81
num_angles_per_block = 1
num_cycles = 3

detector_zero_level = np.min(data)

# Define properties of the optical system

M = 29370.0
aper_rad = 30e-6  # m
focal_length = 3.48e-3  # m
spherical_abe = 2.7e-3  # m
chromatic_abe = 2.6e-3  # m
aper_angle = 0.05e-3  # rad
acc_voltage = 300.0e3  # V
mean_energy_spread = 0.6  # V
defocus = 6e-6  # m

voxel_size = 0.4767e-9  # m


nx = 512
ny = 256 # y is tilt-axis
nz = 350 # z is optical axis

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*(nx/2)*voxel_size,
                                       -rescale_factor*(ny/2)*voxel_size,
                                       -rescale_factor*(nz/2)*voxel_size],
                               max_pt=[rescale_factor*(nx/2)*voxel_size,
                                       rescale_factor*(ny/2)*voxel_size,
                                       rescale_factor*(nz/2)*voxel_size],
                               shape=[nx, ny, nz], dtype='float64')

# Make a 3d single-axis parallel beam geometry with flat detector
#angle_partition = odl.uniform_partition(-62.18*np.pi/180, 58.03*np.pi/180,
#                                        num_angles, nodes_on_bdry=True)
# Make nonuniform angle partition
angle_partition = odl.nonuniform_partition((np.pi/180) * angles, nodes_on_bdry=True)

detector_partition = odl.uniform_partition([-rescale_factor*(nx/2)*voxel_size,
                                            -rescale_factor*(ny/2)*voxel_size],
                                           [rescale_factor*(nx/2)*voxel_size,
                                            rescale_factor*(ny/2)*voxel_size],
                                           [nx, ny])

# The y-axis is the tilt-axis.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=(0, 1, 0),
                                           det_pos_init=(0, 0, 1),
                                           det_axes_init=((1, 0, 0),
                                                          (0, 1, 0)))

# Ray transform
ray_trafo = etomo.BlockRayTransform(reco_space, geometry)

# The image-formation operator models the optics and the detector
# of the electron microscope.
imageFormation_op = etomo.make_imageFormationOp(ray_trafo.range, 
                                                wave_number, spherical_abe,
                                                defocus,
                                                rescale_factor=rescale_factor,
                                                obj_magnitude=obj_magnitude,
                                                abs_phase_ratio=abs_phase_ratio,
                                                aper_rad=aper_rad,
                                                aper_angle=aper_angle,
                                                focal_length=focal_length,
                                                mean_energy_spread=mean_energy_spread,
                                                acc_voltage=acc_voltage,
                                                chromatic_abe=chromatic_abe,
                                                normalize=True)


# Define forward operator as a composition
forward_op = imageFormation_op * ray_trafo


data = forward_op.range.element(np.transpose(data - detector_zero_level,
                                             (2, 0, 1)))
data = etomo.buffer_correction(data)

# %% RECONSTRUCTION

reg_param_list = [3e2, 3e3, 3e4]
gamma_H1_list = [0.8, 0.9, 0.95, 0.99]
Niter_CG_list = [30]

reco_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Experimental/Region1/kaczmarz'

start = time()

for reg_param in reg_param_list:
    print('reg_param= '+str(reg_param))
    for gamma_H1 in gamma_H1_list:
        print('gamma_H1= '+str(gamma_H1))
        for Niter_CG in Niter_CG_list:
            print('NiterCG= '+str(Niter_CG))
            print('time: '+str(timedelta(seconds=time()-start)))

            saveto_path = reco_path+'_gamma_H1='+str(gamma_H1)+'_reg_par='+str(reg_param)+'_niter_CG='+str(Niter_CG)+'_num_cycles='+str(num_cycles)+'/iterate_{}'
            
            callback = odl.solvers.CallbackSaveToDisk(saveto=saveto_path,
                                                      step=num_angles*num_cycles-1,
                                                      impl='numpy')
        
    
            reco = ray_trafo.domain.zero()
            
            kaczmarz_plan = etomo.make_kaczmarz_plan(num_angles,
                                                     block_length=num_angles_per_block,
                                                     method='mls')
            
            ray_trafo_block = ray_trafo.get_sub_operator(kaczmarz_plan[0])
            
            F_post = etomo.make_imageFormationOp(ray_trafo_block.range,
                                                 wave_number, spherical_abe,
                                                 defocus,
                                                 rescale_factor=rescale_factor,
                                                 obj_magnitude=obj_magnitude,
                                                 abs_phase_ratio=abs_phase_ratio,
                                                 aper_rad=aper_rad,
                                                 aper_angle=aper_angle,
                                                 focal_length=focal_length,
                                                 mean_energy_spread=mean_energy_spread,
                                                 acc_voltage=acc_voltage,
                                                 chromatic_abe=chromatic_abe,
                                                 normalize=True)
            

            F_pre = odl.IdentityOperator(reco_space)
            
            
            get_op = etomo.make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre,
                                          Op_post=F_post)
            get_data = etomo.make_data_blocks(data, kaczmarz_plan)
            
            # Optional nonnegativity-constraint
            nonneg_projection = etomo.get_nonnegativity_projection(reco_space)
            
            
            reco = reco_space.zero()
            get_proj_op = etomo.make_Op_blocks(kaczmarz_plan, ray_trafo,
                                               Op_pre=F_pre,
                                               Op_post=None)
            
            etomo.kaczmarz_SART_method(get_proj_op, reco, get_data,
                                       len(kaczmarz_plan),
                                       reg_param*obj_magnitude ** 2,
                                       imageFormationOp=F_post, gamma_H1=gamma_H1,
                                       niter_CG=Niter_CG, callback=callback,
                                       num_cycles=num_cycles,
                                       projection=nonneg_projection)