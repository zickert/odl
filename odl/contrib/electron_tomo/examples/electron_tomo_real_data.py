# -*- coding: utf-8 -*-

"""Phase contrast TEM reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

import os
import odl
from odl.contrib.electron_tomo.block_ray_trafo import BlockRayTransform
from odl.contrib.electron_tomo.kaczmarz_alg import *
from odl.contrib.electron_tomo.image_formation_etomo import *
from odl.contrib.electron_tomo.kaczmarz_util import *
from odl.contrib.electron_tomo.support_constraint import spherical_mask
from odl.contrib.electron_tomo.buffer_correction import buffer_correction
from odl.contrib.electron_tomo.plot_3d import plot_3d_ortho_slices, plot_3d_axis_drive


import matplotlib.pyplot as plt 

from odl.contrib.mrc import FileReaderMRC


dir_path = os.path.abspath('/home/zickert/TEM_reco_project')
file_path_data = os.path.join(dir_path, 'region1.mrc')


with FileReaderMRC(file_path_data) as reader:
    header, data = reader.read()


rescale_factor = 1e9

voxel_size = 0.4767e-9  # m

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
num_angles_per_kaczmarz_block = 1
num_cycles = 3


detector_zero_level = -32768
total_dose = 5000 * 1e18  # total electron dose per m^2
dose_per_img = total_dose / 61
gain = 80  # average nr of digital counts per incident electron


# Define properties of the optical system
# Set focal_length to be the focal_length of the principal (first) lens !
M = 29370.0
aper_rad = 30e-6  # m
focal_length = 3.48e-3  # m
spherical_abe = 2.7e-3  # m
defocus = -6e-6  # m


# Set size of detector pixels (before rescaling to account for magnification)
det_size = 16e-6  # m
det_area = det_size ** 2  # m^2



reco_space = odl.uniform_discr(min_pt=[-rescale_factor*256*voxel_size, -rescale_factor*128*voxel_size,
                                       -rescale_factor*128*voxel_size],
                               max_pt=[rescale_factor*256*voxel_size, rescale_factor*128*voxel_size,
                                       rescale_factor*128*voxel_size],
                               shape=[512, 256, 512],dtype='float64')

angle_partition = odl.uniform_partition(-62.18*np.pi/180, 58.03*np.pi/180, num_angles)
detector_partition = odl.uniform_partition([-rescale_factor*256*voxel_size, -rescale_factor*128*voxel_size],
                                           [rescale_factor*256*voxel_size, rescale_factor*128*voxel_size], [512,256])

# The x-axis is the tilt-axis.
# Check that the geometry matches the one from TEM-simulator!
# In particular, check that det_pos_init and det_axes_init are correct.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=(0, 1, 0),
                                           det_pos_init=(0, 0, 1),
                                           det_axes_init=((1, 0, 0), (0, 1, 0)))
ray_trafo = BlockRayTransform(reco_space, geometry)

imageFormation_op = make_imageFormationOp(ray_trafo.range, 
                                          wave_number, spherical_abe, defocus,
                                          det_size, M, rescale_ctf=True, rescale_ctf_factor=rescale_factor,
                                          obj_magnitude=obj_magnitude,
                                          abs_phase_ratio=abs_phase_ratio,
                                          dose_per_img=dose_per_img, gain=gain,
                                          det_area=det_area)

mask = reco_space.element(spherical_mask, radius=rescale_factor * 1) # * 55e-9)

forward_op = imageFormation_op * ray_trafo * mask
data = forward_op.range.element(np.transpose(data - detector_zero_level, (2, 0, 1)))

data.show(coords=[0, [-2e1, 2e1], [-2e1, 2e1]])

data_bc = buffer_correction(data)

data_bc.show(coords = [0, [-2e1, 2e1], [-2e1, 2e1]])

data_renormalized = data_bc * np.mean(imageFormation_op(imageFormation_op.domain.zero()).asarray())



# %% TRY RECONSTRUCTION

callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())


kaczmarz_plan = make_kaczmarz_plan(num_angles,
                                   num_blocks_per_superblock=num_angles_per_kaczmarz_block,
                                   method='random')

ray_trafo_block = ray_trafo.get_sub_operator(kaczmarz_plan[0])

F_post = make_imageFormationOp(ray_trafo_block.range, wave_number,
                               spherical_abe, defocus, det_size, M,
                               rescale_ctf=True, rescale_ctf_factor=rescale_factor,obj_magnitude=obj_magnitude,
                               abs_phase_ratio=abs_phase_ratio,
                               dose_per_img=dose_per_img, gain=gain,
                               det_area=det_area)

F_pre = odl.MultiplyOperator(mask, reco_space, reco_space)

get_op = make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre, Op_post=F_post)
get_data = make_data_blocks(data_renormalized, kaczmarz_plan)

# Optional nonnegativity-constraint
nonneg_constraint = odl.solvers.IndicatorNonnegativity(reco_space).proximal(1)


def nonneg_projection(x):
    x[:] = nonneg_constraint(x)

reco = reco_space.zero()
get_proj_op = make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre,
                             Op_post=None)

kaczmarz_SART_method(get_proj_op, reco, get_data, len(kaczmarz_plan),
                     regpar*obj_magnitude ** 2,
                     imageFormationOp=F_post, gamma_H1=0.9, niter_CG=30,
                     callback=callback, num_cycles=num_cycles, projection=nonneg_projection)


# Plot results
plot_3d_ortho_slices(reco)



# %%
for angle_idx in range(num_angles):
    proj_op = ray_trafo.get_sub_operator([angle_idx]);
    proj = proj_op(proj_op.domain.one());
    center = 0.5*(proj.space.min_pt[0] + proj.space.max_pt[0]);
    proj.show(coords = [center,None,None])
    
    
# %%
for angle_idx in range(num_angles):
    image = proj_op.range.element(data_renormalized.asarray()[angle_idx,:,:])
    center = 0.5*(image.space.min_pt[0] + image.space.max_pt[0]);
    image.show(coords = [center,None,None])
    #plt.figure(); plt.imshow(image, vmin = 0.9, vmax = 1.0); plt.colorbar();

