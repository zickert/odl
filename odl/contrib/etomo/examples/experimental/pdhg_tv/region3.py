"""Electron tomography reconstruction example using real data."""

import numpy as np
import os
import odl
from odl.contrib import etomo
import matplotlib.pyplot as plt 
from odl.contrib.mrc import FileReaderMRC

# Read data
dir_path = os.path.abspath('/home/zickert/TEM_reco_project/Data/Experimental')
file_path_data = os.path.join(dir_path, 'region3.mrc')

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
                                        num_angles)

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
                                                abs_phase_ratio=abs_phase_ratio)

# Define a spherical mask to implement support constraint.
mask = reco_space.element(etomo.spherical_mask, radius=rescale_factor * 1) # * 55e-9)

# Define forward operator as a composition
forward_op = imageFormation_op * ray_trafo * mask
data = forward_op.range.element(np.transpose(data - detector_zero_level,
                                             (2, 0, 1)))

data.show(coords=[0, None, None])

data_bc = etomo.buffer_correction(data)

data_bc.show(coords=[0, None, None])

data_renormalized = data_bc * np.mean(imageFormation_op(imageFormation_op.domain.zero()).asarray())



#PDHG
####################
# --- Set up the inverse problem --- #

# Initialize gradient operator
gradient = odl.Gradient(reco_space)

# Column vector of two operators
op = odl.BroadcastOperator(forward_op, gradient)

# Do not use the g functional, set it to zero.
g = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(forward_op.range).translated(data_renormalized)

#
reg_param = 1000

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = reg_param * odl.solvers.GroupL1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * 7.4465020879509245 # 1.1 * odl.power_method_opnorm(op)

niter = 200  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable



# Choose a starting point
x = reco_space.zero()

# define callback 
callback = (odl.solvers.CallbackPrintIteration(step=10) &
            odl.solvers.CallbackShow(step=10))
# Run the algorithm
odl.solvers.pdhg(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
                 callback=callback)


# -*- coding: utf-8 -*-

