"""Electron tomography reconstruction example using real data."""

import numpy as np
import os
import odl
from odl.contrib import etomo
import matplotlib.pyplot as plt 
from odl.contrib.mrc import FileReaderMRC

# Read data
dir_path = os.path.abspath('/mnt/imagingnas/data/Users/gzickert/TEM/Data/Experimental')
file_path_data = os.path.join(dir_path, 'region1.mrc')

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

#dose = 1 # not known
#det_after_M = (det_size / M)  **2
#
#electrons_per_pixel = (dose / det_after_M ) / 61



detector_zero_level = np.min(data)

# Define properties of the optical system
# Set focal_length to be the focal_length of the principal (first) lens !
# M = 29370.0
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
# Angles: uniformly spaced, n = num_angles, min = -62.18 deg, max = 58.03 deg
angle_partition = odl.uniform_partition(-62.18*np.pi/180, 58.03*np.pi/180,
                                        num_angles, nodes_on_bdry=True)

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
                                                chromatic_abe=chromatic_abe)


# Define a spherical mask to implement support constraint.
mask = reco_space.element(etomo.spherical_mask, radius=rescale_factor * 1) # * 55e-9)

# Define forward operator as a composition
forward_op = imageFormation_op * ray_trafo * mask
data = forward_op.range.element(np.transpose(data - detector_zero_level,
                                             (2, 0, 1)))


lin_op = forward_op(reco_space.zero()) + forward_op.derivative(reco_space.zero())

forward_op = lin_op

#data.show(coords=[0, None, None])

data_bc = etomo.buffer_correction(data)

#data_bc.show(coords=[0, None, None])

data_renormalized = data_bc * np.mean(forward_op(forward_op.domain.zero()).asarray())

data = data_renormalized


reg_par_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
gamma_huber_list = [1e-2, 1e-3, 1e-1]


reco_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Experimental/Region1/gradient_descent_huber_reg_LINEARIZED'

for gamma_huber in gamma_huber_list:
    for reg_par in reg_par_list:
    
        saveto_path = reco_path+'_gamma='+str(gamma_huber)+'_reg_par='+str(reg_par)+'/iterate_{}'
        
        callback = odl.solvers.CallbackSaveToDisk(saveto=saveto_path,
                                                  step=1000, impl='numpy')
    
        reco = reco_space.zero()
    
        nonneg_projection = etomo.get_nonnegativity_projection(reco_space)
    
    
        gradient = odl.Gradient(reco_space)
        huber_func = odl.solvers.Huber(gradient.range, gamma=gamma_huber)
        TV_smothened = huber_func * gradient
    
        # l2-squared data matching
        l2_norm = odl.solvers.L2NormSquared(forward_op.range).translated(data)
    
        # reg_par = 0.01
    
        f = l2_norm * forward_op + reg_par * TV_smothened
    
        ls = odl.solvers.BacktrackingLineSearch(f)
    
        odl.solvers.steepest_descent(f, reco, line_search=ls, maxiter=1001, callback=callback,
                                     projection=nonneg_projection)
