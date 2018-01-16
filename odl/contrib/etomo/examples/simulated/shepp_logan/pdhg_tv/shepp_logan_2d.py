"""Electron tomography reconstruction example in 2d."""


import numpy as np
import matplotlib.pyplot as plt
import odl
from odl.contrib import etomo


obj_magnitude = 1e-2

# Relative noise level
noise_lvl = 1e-1

regpar = 1e1

num_angles = 360
num_angles_per_block = 1
num_cycles = 1

wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length

aper_rad = 0.5*40e-6  # m
focal_length = 2.7e-3  # m
spherical_abe = 2.1e-3  # m
defocus = 3e-6  # m

# In this toy example, rescale_factor can be chosen arbitrarily, but 0.5e9 will
# give a particle of roughly the same size as the rna_phantom from the
# TEM-Simulator. A greater value of this factor means that the corresponding
# 'true particle' is smaller, hence varies on a larger scale in frequency-space
# This in turn means that the CTF will have a greater effect, leading to more
# 'fringes' in the data.
rescale_factor = 0.5e9

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(min_pt=[-20] * 2, max_pt=[20] * 2,
                               shape=[300] * 2)

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = num_angles, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, num_angles)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform
ray_trafo = etomo.BlockRayTransform(reco_space, geometry)

# The image-formation operator models the optics and the detector
# of the electron microscope.
imageFormation_op = etomo.make_imageFormationOp(ray_trafo.range,
                                                wave_number, spherical_abe,
                                                defocus,
                                                rescale_factor=rescale_factor,
                                                obj_magnitude=obj_magnitude)

# Define a spherical mask to implement support constraint.
mask = reco_space.element(etomo.spherical_mask, radius=19)

# Define forward operator as a composition
forward_op = imageFormation_op * ray_trafo * mask
# forward_op_linearized = forward_op.derivative(reco_space.zero())

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)  # (1+1j) *

# Create data by calling the forward operator on the phantom
data = forward_op(phantom)

# Add noise to the data
noise = odl.phantom.white_noise(data.space)
data += (noise_lvl * (data.space.one()-data).norm() / noise.norm()) * noise

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
l2_norm = odl.solvers.L2NormSquared(forward_op.range).translated(data)

#
reg_param = 0.003

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = reg_param * odl.solvers.GroupL1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * 0.203287581820518185 # 1.1 * odl.power_method_opnorm(op)

niter = 10000  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable



# Choose a starting point
x = reco_space.zero()

# define callback 
callback = (odl.solvers.CallbackPrintIteration(step=200) &
            odl.solvers.CallbackShow(step=200))
# Run the algorithm
odl.solvers.pdhg(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
                 callback=callback)



