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
file_path_data = os.path.join(dir_path, 'region1.mrc')
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
# Set focal_length to be the focal_length of the principal (first) lens !
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

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*256*voxel_size,
                                       -rescale_factor*128*voxel_size,
                                       -rescale_factor*256*voxel_size],
                               max_pt=[rescale_factor*256*voxel_size,
                                       rescale_factor*128*voxel_size,
                                       rescale_factor*256*voxel_size],
                               shape=[512, 256, 512], dtype='float64')

# Make a 3d single-axis parallel beam geometry with flat detector
#angle_partition = odl.uniform_partition(-62.18*np.pi/180, 58.03*np.pi/180,
#                                        num_angles, nodes_on_bdry=True)
# Make nonuniform angle partition
angle_partition = odl.nonuniform_partition((np.pi/180) * angles, nodes_on_bdry=True)

detector_partition = odl.uniform_partition([-rescale_factor*256*voxel_size,
                                            -rescale_factor*128*voxel_size],
                                           [rescale_factor*256*voxel_size,
                                            rescale_factor*128*voxel_size],
                                           [512, 256])

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
#%%

reg_param_list = [1e-4, 3e-4, 1e-3, 3e-3]
step_param_list = [1e-3, 1e-2, 1e-1]
niter = 1001  # Number of iterations
steps_to_save = 1000

reco_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Experimental/Region1/pdhg_tv_pos_constr'

start = time()

for reg_param in reg_param_list:
    print('reg_param='+str(reg_param))
    for step_param in step_param_list:
        print('step_param='+str(step_param))
        print('time: '+str(timedelta(seconds=time()-start)))

    
        saveto_path = reco_path+'/step_par='+str(step_param)+'_reg_par='+str(reg_param)+'_iterate_{}'
        
        callback = odl.solvers.CallbackSaveToDisk(saveto=saveto_path,
                                                  step=steps_to_save, impl='numpy')
    

        # Initialize gradient operator
        gradient = odl.Gradient(reco_space)
        
        # Column vector of two operators
        op = odl.BroadcastOperator(forward_op, gradient)
        
        # Do not use the g functional, set it to zero.
        g = odl.solvers.IndicatorNonnegativity(reco_space)
        
        # Create functionals for the dual variable
        
        # l2-squared data matching
        l2_norm = odl.solvers.L2NormSquared(forward_op.range).translated(data)
        
        # Isotropic TV-regularization i.e. the l1-norm
        l1_norm = reg_param * odl.solvers.GroupL1Norm(gradient.range)
        
        # Combine functionals, order must correspond to the operator K
        f = odl.solvers.SeparableSum(l2_norm, l1_norm)
        
        # --- Select solver parameters and solve using PDHG --- #
        
        # Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
        op_norm = 1.1 * 0.024448060645880614 # 1.1 * odl.power_method_opnorm(forward_op.derivative(reco_space.zero()))

        tau = step_param / op_norm  # Step size for the primal variable
        sigma = step_param / op_norm  # Step size for the dual variable

        # Choose a starting point
        x = reco_space.zero()
        
        # Run the algorithm
        odl.solvers.pdhg(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
                         callback=callback)


