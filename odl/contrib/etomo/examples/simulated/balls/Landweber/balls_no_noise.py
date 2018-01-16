"""Electron tomography reconstruction example using data from TEM-Simulator"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import os
import odl
from odl.contrib import etomo
from odl.contrib.mrc import FileReaderMRC

# Read phantom and data.
dir_path = os.path.abspath('/home/zickert/TEM_reco_project/Data/Simulated/Balls/No_noise')
file_path_phantom = os.path.join(dir_path, 'balls_phantom.mrc')
file_path_tiltseries = os.path.join(dir_path, 'tiltseries.mrc')

with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
with FileReaderMRC(file_path_tiltseries) as tiltseries_reader:
    tiltseries_header, data_asarray = tiltseries_reader.read()

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
magic_factor = 1
obj_magnitude = magic_factor * sigma / rescale_factor
num_angles = 61
num_angles_per_block = 1
num_cycles = 3

# Define sample diameter and height. We take flat sample
sample_diam = 1200e-9  # m
sample_height = 150e-9  # m

# Define properties of the optical system
# Set focal_length to be the focal_length of the principal (first) lens !
M = 25000.0
aper_rad = 0.5*40e-6  # m
focal_length = 2.7e-3  # m
spherical_abe = 2.1e-3  # m
defocus = 3e-6  # m

# Set size of detector pixels (before rescaling to account for magnification)
det_size = 16e-6  # m

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
angle_partition = odl.uniform_partition(-np.pi/3, np.pi/3, num_angles)
detector_partition = odl.uniform_partition([-rescale_factor*det_size/M * 210/2,
                                            -rescale_factor*det_size/M * 250/2],
                                           [rescale_factor*det_size/M * 200/2,
                                            rescale_factor*det_size/M * 250/2],
                                            [210, 250])

# The x-axis is the tilt-axis.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=(1, 0, 0),
                                           det_pos_init=(0, 0, -1),
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
                                                abs_phase_ratio=abs_phase_ratio)

# Define forward operator as a composition
forward_op = imageFormation_op * ray_trafo
phantom = reco_space.element(phantom_asarray)

# remove background
bg_cst = np.min(phantom)
phantom -= bg_cst

# Create data by calling the forward operator on the phantom
data_from_this_model = forward_op(phantom)

# Make  a ODL discretized function of the MRC data
data = forward_op.range.element(np.transpose(data_asarray, (2, 0, 1)))

# Correct for diffrent pathlenght of the electrons through the buffer
data = etomo.buffer_correction(data)
data_from_this_model = etomo.buffer_correction(data_from_this_model)

# Plot corrected data
data_from_this_model.show(coords=[0, None, None])
data.show(coords=[0, None, None])

# Renormalize data so that it matches "data_from_this_model"
data *= np.mean(data_from_this_model.asarray())

# Plot results
# etomo.plot_3d_ortho_slices(phantom)
# etomo.plot_3d_ortho_slices(reco)

## Save planes of reco (orthogonal to x,y and z axes)
#nn_reco_fig_x = reco.show(title='balls_no_noise_reco_x',
#                          coords=[0, None, None])
#nn_reco_fig_x.savefig('balls_no_noise_reco_x')
#
#nn_reco_fig_y = reco.show(title='balls_no_noise_reco_y',
#                          coords=[None, 0, None])
#nn_reco_fig_y.savefig('balls_no_noise_reco_y')
#
#nn_reco_fig_z = reco.show(title='balls_no_noise_reco_z',
#                          coords=[None, None, 0])
#nn_reco_fig_z.savefig('balls_no_noise_reco_z')


#%% 
reco = ray_trafo.domain.zero()
reco = phantom
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

#Landweber iterations
nonneg_projection = etomo.get_nonnegativity_projection(reco_space)

odl.solvers.landweber(forward_op, reco, data, 1000, omega=3e1, callback=callback,projection=nonneg_projection)
