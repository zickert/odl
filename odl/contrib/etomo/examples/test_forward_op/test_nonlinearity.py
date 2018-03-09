"""Electron tomography reconstruction example using data from TEM-Simulator"""


import matplotlib.pyplot as plt 
import numpy as np
import os
import odl
from odl.contrib import etomo
from odl.contrib.mrc import (
    FileReaderMRC, FileWriterMRC, mrc_header_from_params)

# Read phantom and data.
dir_path = os.path.abspath('/mnt/imagingnas/data/Users/gzickert/TEM/Data/Simulated/Balls/No_noise')
file_path_phantom = os.path.join(dir_path, 'balls_phantom.mrc')
file_path_phantom_im = os.path.join(dir_path, 'balls_phantom_im.mrc')
file_path_tiltseries = os.path.join(dir_path, 'tiltseries_perfect_mtf_perfect_dqe.mrc')

with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
with FileReaderMRC(file_path_phantom_im) as phantom_im_reader:
    phantom_im_header, phantom_im_asarray = phantom_im_reader.read()
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
total_dose = 5000e18
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
                               shape=[210, 250, 40], dtype='float32')
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
                                                mtf_a=0.7, mtf_b=0.2,
                                                mtf_c=0.1, mtf_alpha=10,
                                                mtf_beta=40,
                                                magnification=25000,
                                                dose_per_img=dose_per_img,
                                                gain=gain,
                                                det_area=det_area,
                                                ice_thickness=ice_thickness,
                                                sigma=sigma)

phantom = reco_space.element(phantom_asarray)
phantom_im = reco_space.element(phantom_im_asarray)


# Define forward operator as a composition
forward_op = imageFormation_op * ray_trafo
lin_op = forward_op(reco_space.zero())+forward_op.derivative(reco_space.zero())

# remove background
bg_cst = np.min(phantom)
phantom -= bg_cst

# Create data by calling the forward operator on the phantom
data_from_this_model = forward_op(phantom)
data_from_this_model_lin = lin_op(phantom)

# Make  a ODL discretized function of the MRC data
data = forward_op.range.element(np.transpose(data_asarray, (2, 0, 1)))

# Correct for diffrent pathlenght of the electrons through the buffer
#data = etomo.buffer_correction(data, coords=[[0, 0.1], [0, 0.1]])
#data_from_this_model = etomo.buffer_correction(data_from_this_model,
#                                               coords=[[0, 0.1], [0, 0.1]])
#data_from_this_model_lin = etomo.buffer_correction(data_from_this_model_lin,
#                                                   coords=[[0, 0.1], [0, 0.1]])


nonlinearity = data_from_this_model-data_from_this_model_lin
mismatch = data-data_from_this_model
mismatch_lin = data-data_from_this_model_lin 
# Renormalize data so that it matches "data_from_this_model"
#data *= np.mean(data_from_this_model_lin.asarray())

#%%
coords = [0, None, None]
coords = [0, [-62,-58], [-62,-58]]

(data).show(coords=coords, title='TEM-Simulator data')
(data_from_this_model).show(coords=coords, title='data from my op')
(data_from_this_model_lin).show(coords=coords, title='data from my lin op')
nonlinearity.show(coords=coords, title='nonlinearity')
mismatch.show(coords=coords, title='mismatch')
mismatch_lin.show(coords=coords, title='mismatch_lin')

#print(str((data).norm()))
#print(str(mismatch.norm()))
#print(str(mismatch_lin.norm()))
#
#
#data_zero_tilt = (data).asarray()[30,:,:]
#data_max_tilt = (data).asarray()[60,:,:]
#mismatch_zero_tilt = mismatch.asarray()[30,:,:]
#mismatch_max_tilt = mismatch.asarray()[60,:,:]
#
#print(str(np.linalg.norm(data_zero_tilt)))
#print(str(np.linalg.norm(mismatch_zero_tilt)))
#print(str(np.linalg.norm(data_max_tilt)))
#print(str(np.linalg.norm(mismatch_max_tilt)))
#
#mismatch_zero_tilt_cropped = mismatch_zero_tilt[:30,:30]
#mismatch_max_tilt_cropped = mismatch_max_tilt[:30,:30]
#
#plt.figure()
#plt.imshow(mismatch_zero_tilt)
#plt.colorbar()
#plt.figure()
#plt.imshow(mismatch_max_tilt)
#plt.colorbar()
#
#plt.figure()
#plt.imshow(mismatch_zero_tilt_cropped)
#plt.colorbar()
#plt.figure()
#plt.imshow(mismatch_max_tilt_cropped)
#plt.colorbar()