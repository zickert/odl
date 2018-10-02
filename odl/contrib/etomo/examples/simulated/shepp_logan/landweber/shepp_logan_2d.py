"""Electron tomography reconstruction example in 2d."""


import numpy as np
import odl
from odl.contrib import etomo

abs_phase_ratio = 0.1

obj_magnitude = 1e-2

# Relative noise level
noise_lvl = 1e-2

regpar = 1e1

num_angles = 360
num_angles_per_block = 1
num_cycles = 1

wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length

aper_rad = 0.5*40e-6  # m
focal_length = 2.7e-3  # m
spherical_abe = 2.1e-3  # m
chromatic_abe = 2.2e-3  # m
defocus = 3e-6  # m
aper_angle = 0.1e-3  # rad
acc_voltage = 200.0e3  # V
mean_energy_spread = 1.3  # V


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
angle_partition = odl.uniform_partition(-np.pi/3, np.pi/3, num_angles)
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
                                                obj_magnitude=obj_magnitude,
                                                abs_phase_ratio=abs_phase_ratio,
                                                aper_rad=aper_rad,
                                                aper_angle=aper_angle,
                                                focal_length=focal_length,
                                                mean_energy_spread=mean_energy_spread,
                                                acc_voltage=acc_voltage,
                                                chromatic_abe=chromatic_abe,
                                                normalize=True)

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

# %%

reco = reco_space.zero()
# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())
# Optional nonnegativity-constraint
nonneg_projection = etomo.get_nonnegativity_projection(reco_space)

#Landweber iterations
odl.solvers.landweber(forward_op, reco, data, 1000, omega=3e1, callback=callback,projection=nonneg_projection)
