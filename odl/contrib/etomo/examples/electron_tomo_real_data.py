"""Electron tomography reconstruction example using real data."""

import numpy as np
import os
import odl
from odl.contrib import etomo
import matplotlib.pyplot as plt 
from odl.contrib.mrc import FileReaderMRC

# Read data
dir_path = os.path.abspath('/home/zickert/TEM_reco_project')
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
wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length
sigma = e_mass * e_charge / (wave_number * planck_bar ** 2)

abs_phase_ratio = 0.5
obj_magnitude = 0.5*sigma / rescale_factor
regpar = 3e3
num_angles = 81
num_angles_per_block = 1
num_cycles = 3

detector_zero_level = -32768

# Define properties of the optical system
# Set focal_length to be the focal_length of the principal (first) lens !
# M = 29370.0
aper_rad = 30e-6  # m
focal_length = 3.48e-3  # m
spherical_abe = 2.7e-3  # m
defocus = -6e-6  # m

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

data.show(coords=[0, [-2e1, 2e1], [-2e1, 2e1]])

data_bc = etomo.buffer_correction(data)

data_bc.show(coords=[0, [-2e1, 2e1], [-2e1, 2e1]])

data_renormalized = data_bc * np.mean(imageFormation_op(imageFormation_op.domain.zero()).asarray())


# %% TRY RECONSTRUCTION

callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())


kaczmarz_plan = etomo.make_kaczmarz_plan(num_angles,
                                         block_length=num_angles_per_block,
                                         method='random')

ray_trafo_block = ray_trafo.get_sub_operator(kaczmarz_plan[0])

F_post = etomo.make_imageFormationOp(ray_trafo_block.range, wave_number,
                                     spherical_abe, defocus,
                                     rescale_factor=rescale_factor,
                                     obj_magnitude=obj_magnitude,
                                     abs_phase_ratio=abs_phase_ratio)

F_pre = odl.MultiplyOperator(mask, reco_space, reco_space)

get_op = etomo.make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre,
                              Op_post=F_post)
get_data = etomo.make_data_blocks(data_renormalized, kaczmarz_plan)

# Optional nonnegativity-constraint
nonneg_constraint = odl.solvers.IndicatorNonnegativity(reco_space).proximal(1)


def nonneg_projection(x):
    x[:] = nonneg_constraint(x)


reco = reco_space.zero()
get_proj_op = etomo.make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre,
                                   Op_post=None)

etomo.kaczmarz_SART_method(get_proj_op, reco, get_data, len(kaczmarz_plan),
                           regpar*obj_magnitude ** 2,
                           imageFormationOp=F_post, gamma_H1=0.9, niter_CG=30,
                           callback=callback, num_cycles=num_cycles,
                           projection=nonneg_projection)


# Plot results
etomo.plot_3d_ortho_slices(reco)


# %%
for angle_idx in range(num_angles):
    proj_op = ray_trafo.get_sub_operator([angle_idx])
    proj = proj_op(proj_op.domain.one())
    center = 0.5*(proj.space.min_pt[0] + proj.space.max_pt[0])
    proj.show(coords=[center, None, None])

# %%
for angle_idx in range(num_angles):
    image = proj_op.range.element(data_renormalized.asarray()[angle_idx, :, :])
    center = 0.5*(image.space.min_pt[0] + image.space.max_pt[0])
    image.show(coords=[center, None, None])
    # plt.figure(); plt.imshow(image, vmin = 0.9, vmax = 1.0); plt.colorbar();
