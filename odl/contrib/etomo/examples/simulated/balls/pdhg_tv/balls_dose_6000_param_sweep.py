"""Electron tomography reconstruction example using data from TEM-Simulator"""

import matplotlib.pyplot as plt 
import numpy as np
import os
import odl
from odl.contrib import etomo
from odl.contrib.mrc import FileReaderMRC
from time import time
from datetime import timedelta

# Read phantom and data.
dir_path = os.path.abspath('/mnt/imagingnas/data/Users/gzickert/TEM/Data/Simulated/Balls/dose_6000')
file_path_phantom = os.path.join(dir_path, 'balls_phantom.mrc')
#file_path_tiltseries = os.path.join(dir_path, 'tiltseries_perfect_mtf_perfect_dqe.mrc')

with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
#with FileReaderMRC(file_path_tiltseries) as tiltseries_reader:
#    tiltseries_header, data_asarray = tiltseries_reader.read()

# The reconstruction space will be rescaled according to rescale_factor in
# order to avoid numerical issues related to having a very small reco space.
rescale_factor = 1e9

#  Define some physical constants
e_mass = 9.11e-31  # kg
e_charge = 1.602e-19  # C
planck_bar = 1.059571e-34  # Js/rad

wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length

sigma = e_mass * e_charge / (wave_number * planck_bar ** 2)

abs_phase_ratio = 0.1
obj_magnitude = sigma / rescale_factor
num_angles = 61

ice_thickness = 50e-9

# Define properties of the optical system
# Set focal_length to be the focal_length of the principal (first) lens !
M = 25000.0
aper_rad = 0.5*40e-6  # m
focal_length = 2.7e-3  # m
spherical_abe = 2.1e-3  # m
chromatic_abe = 2.2e-3  # m
aper_angle = 0.1e-3  # rad
acc_voltage = 200.0e3  # V
mean_energy_spread = 1.3  # V
defocus = 3e-6  # m
gain = 80
total_dose = 6000e18  # m^-2
dose_per_img = total_dose / num_angles

# Set size of detector pixels (before rescaling to account for magnification)
det_size = 16e-6  # m
det_area = det_size ** 2

# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*210e-9/4,
                                       -rescale_factor*250e-9/4,
                                       -rescale_factor*40e-9/4],
                               max_pt=[rescale_factor*210e-9/4,
                                       rescale_factor*250e-9/4,
                                       rescale_factor*40e-9/4],
                               shape=[210, 250, 40], dtype='float64')
# Make a 3d single-axis parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 180, min = 0, max = pi
angle_partition = odl.uniform_partition(-np.pi/3, np.pi/3, num_angles,
                                        nodes_on_bdry=True)
detector_partition = odl.uniform_partition([-rescale_factor*det_size/M * 210/2,
                                            -rescale_factor*det_size/M * 250/2],
                                           [rescale_factor*det_size/M * 210/2,
                                            rescale_factor*det_size/M * 250/2],
                                            [210, 250])

# The x-axis is the tilt-axis.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=(1, 0, 0),
                                           det_pos_init=(0, 0, -1),
                                           det_axes_init=((1, 0, 0),
                                                          (0, 1, 0)))

# Ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

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

phantom = reco_space.element(phantom_asarray)

# We remove the background from the phantom
bg_cst = np.min(phantom)
phantom -= bg_cst

# Create data by calling forward op on phantom
data = forward_op(phantom)

# Add noise
total_cst = dose_per_img
total_cst *= (1 / M) ** 2
total_cst *= det_area
buffer_contr = data.space.element(etomo.buffer_contribution,
                                  sigma=sigma,
                                  ice_thickness=ice_thickness)
data = total_cst * buffer_contr * data
data = gain * odl.phantom.poisson_noise(data)

# Correct for different path-lengths through the buffer
data = etomo.buffer_correction(data, coords=[[0, 0.1], [0, 0.1]])

# %%

#PDHG
####################
# --- Set up the inverse problem --- #

reg_param_list = [3e-4]
step_param_list = [1e-1]
niter = 10001  # Number of iterations
steps_to_save = 1000

reco_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/dose_6000/pdhg_tv_pos_constr'

start = time()


for reg_param in reg_param_list:
    print('reg_param='+str(reg_param))
    for step_param in step_param_list:
        print('step_param='+str(step_param))
        print('time: '+str(timedelta(seconds=time()-start)))

        saveto_path = reco_path+'/step_par='+str(step_param)+'_reg_par='+str(reg_param)+'_iterate_{}'
        
        callback = odl.solvers.CallbackSaveToDisk(saveto=saveto_path,
                                                  step=steps_to_save,
                                                  impl='numpy')
    
        
        
        # Initialize gradient operator
        gradient = odl.Gradient(reco_space)
        
        # Column vector of two operators
        op = odl.BroadcastOperator(forward_op, gradient)
        
        # Do not use the g functional, set it to zero.
        g = odl.solvers.IndicatorNonnegativity(reco_space)
        
        # Create functionals for the dual variable
        
        # l2-squared data matching
        l2_norm = odl.solvers.L2NormSquared(forward_op.range).translated(data)
        
        #
        #reg_param = 0.015
        
        # Isotropic TV-regularization i.e. the l1-norm
        l1_norm = reg_param * odl.solvers.GroupL1Norm(gradient.range)
        
        # Combine functionals, order must correspond to the operator K
        f = odl.solvers.SeparableSum(l2_norm, l1_norm)
        
        # --- Select solver parameters and solve using PDHG --- #
        
        # Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
        op_norm = 1.1 * 0.073 # 1.1 * odl.power_method_opnorm(forward_op.derivative(reco_space.zero()))
        
        

        tau = step_param / op_norm  # Step size for the primal variable
        sigma = step_param / op_norm  # Step size for the dual variable
        
        
        
        # Choose a starting point
        x = reco_space.zero()
        #x = 0.5*phantom
        
        # Run the algorithm
        odl.solvers.pdhg(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
                         callback=callback)


