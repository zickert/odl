"""Phase contrast TEM reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

import os
import odl
from odl.contrib import etomo

import matplotlib.pyplot as plt 

from odl.contrib.mrc import FileReaderMRC

dir_path = os.path.abspath('/home/zickert/TEM_reco_project/One_particle_new_simulation')
file_path_phantom = os.path.join(dir_path, 'rna_phantom.mrc')
file_path_phantom_abs = os.path.join(dir_path, 'rna_phantom_abs.mrc')
file_path_map = os.path.join(dir_path, '1I3Q_map.mrc')
file_path_map_abs = os.path.join(dir_path, '1I3Q_abs_map.mrc')
file_path_tiltseries = os.path.join(dir_path, 'tiltseries.mrc')
file_path_tiltseries_nonoise = os.path.join(dir_path, 'tiltseries_nonoise.mrc')

with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
with FileReaderMRC(file_path_phantom_abs) as phantom_abs_reader:
    phantom_abs_header, phantom_abs_asarray = phantom_abs_reader.read()
with FileReaderMRC(file_path_map) as map_reader:
    map_header, map_asarray = map_reader.read()
with FileReaderMRC(file_path_map_abs) as map_abs_reader:
    map_abs_header, map_abs_asarray = map_abs_reader.read()
with FileReaderMRC(file_path_tiltseries) as tiltseries_reader:
    tiltseries_header, data_asarray = tiltseries_reader.read()
with FileReaderMRC(file_path_tiltseries_nonoise) as tiltseries_nonoise_reader:
    tiltseries_nonoise_header, data_nonoise_asarray = tiltseries_nonoise_reader.read()


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
regpar = 2e3
num_angles = 61
num_angles_per_kaczmarz_block = 1
num_cycles = 3


total_dose = 5000 * 1e18  # total electron dose per m^2
dose_per_img = total_dose / 61
gain = 80  # average nr of digital counts per incident electron

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

# Define constants defining the modulation transfer function. (a,b,c) = (0,0,1)
# corresponds to det_op = identity_op
mtf_a = 0
mtf_b = 0
mtf_c = 1
mtf_alpha = 10
mtf_beta = 40

# Set size of detector pixels (before rescaling to account for magnification)
det_size = 16e-6  # m
det_area = det_size ** 2  # m^2

reco_space = odl.uniform_discr(min_pt=[-rescale_factor*95e-9/4,
                                       -rescale_factor*100e-9/4,
                                       -rescale_factor*80e-9/4],
                               max_pt=[rescale_factor*95e-9/4,
                                       rescale_factor*100e-9/4,
                                       rescale_factor*80e-9/4],
                               shape=[95, 100, 80], dtype='float64')

angle_partition = odl.uniform_partition(-np.pi/3, np.pi/3, num_angles)
detector_partition = odl.uniform_partition([-rescale_factor*det_size/M * 200/2] * 2,
                                           [rescale_factor*det_size/M * 200/2] * 2, [200] * 2)

# The x-axis is the tilt-axis.
# Check that the geometry matches the one from TEM-simulator!
# In particular, check that det_pos_init and det_axes_init are correct.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=(1, 0, 0),
                                           det_pos_init=(0, 0, -1),
                                           det_axes_init=((1, 0, 0),
                                                          (0, 1, 0)))
ray_trafo = etomo.BlockRayTransform(reco_space, geometry)

imageFormation_op = etomo.make_imageFormationOp(ray_trafo.range, 
                                                wave_number, spherical_abe,
                                                defocus, det_size, M,
                                                rescale_ctf=True,
                                                rescale_ctf_factor=rescale_factor,
                                                obj_magnitude=obj_magnitude,
                                                abs_phase_ratio=abs_phase_ratio,
                                                dose_per_img=dose_per_img, gain=gain,
                                                det_area=det_area)

mask = reco_space.element(etomo.spherical_mask,
                          radius=rescale_factor * 10.0e-9)

# Leave out detector operator for simplicity
forward_op = imageFormation_op * ray_trafo

phantom = reco_space.element(phantom_asarray)

phantom_abs = reco_space.element(phantom_abs_asarray)


# remove background
bg_cst = np.min(phantom)

phantom -= bg_cst


data = forward_op(phantom)

true_data = forward_op.range.element(np.transpose(data_asarray,
                                                  (2, 0, 1)))


reco = ray_trafo.domain.zero()

phantom.show(coords=[0, None, None])
phantom_abs.show(coords=[0, None, None])
data.show(coords=[0, [-2e1, 2e1], [-2e1, 2e1]])

#plt.imshow(true_data, cmap='gray')
#plt.colorbar()
true_data.show(coords=[0, [-2e1, 2e1], [-2e1, 2e1]])

#background = bg_cst * ray_trafo.domain.one()
#
#bg_abs_cst = np.min(phantom_abs)
#bg_ratio = bg_abs_cst/bg_cst
#
#imageFormation_op_bg = make_imageFormationOp(ray_trafo.range, 
#                                          wave_number, spherical_abe, defocus,
#                                          det_size, M, rescale_ctf=False,
#                                          obj_magnitude=obj_magnitude,abs_phase_ratio=bg_ratio,
#                                          dose_per_img=dose_per_img, gain=gain,
#                                          det_area=det_area)
#
#forward_op_bg = imageFormation_op_bg * ray_trafo 
#
#data_bg = forward_op_bg(background)
#data_bg.show(coords=[0,None,None])

true_data_bc = etomo.buffer_correction(true_data)
data_bc = etomo.buffer_correction(data)

data_bc.show(coords=[0, [-2e1, 2e1], [-2e1, 2e1]])
true_data_bc.show(coords=[0, [-2e1, 2e1], [-2e1, 2e1]])

data_renormalized = true_data_bc * (np.mean(data.asarray()))


# %% TRY RECONSTRUCTION

callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())


kaczmarz_plan = etomo.make_kaczmarz_plan(num_angles,
                                         num_blocks_per_superblock=num_angles_per_kaczmarz_block,
                                         method='random')

ray_trafo_block = ray_trafo.get_sub_operator(kaczmarz_plan[0])

F_post = etomo.make_imageFormationOp(ray_trafo_block.range, wave_number,
                                     spherical_abe, defocus, det_size, M,
                                     rescale_ctf=True,
                                     rescale_ctf_factor=rescale_factor,
                                     obj_magnitude=obj_magnitude,
                                     abs_phase_ratio=abs_phase_ratio,
                                     dose_per_img=dose_per_img, gain=gain,
                                     det_area=det_area)

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
etomo.plot_3d_ortho_slices(phantom)
etomo.plot_3d_ortho_slices(reco)


# %%


#map_slice = map_asarray[:,:,79]
#map_abs_slice = map_abs_asarray[:,:,79]
#                                
#plt.imshow(map_slice)
#plt.colorbar()
#plt.figure()
#plt.imshow(map_abs_slice)
#plt.colorbar()                               