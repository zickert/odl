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
obj_magnitude = sigma / rescale_factor
num_angles = 61

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

# Set size of detector pixels (before rescaling to account for magnification)
det_size = 16e-6  # m

det_size /=10

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
                                                keep_real=True)

phantom = reco_space.element(phantom_asarray)

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
data_from_this_model = etomo.buffer_correction(data_from_this_model,
                                               coords=[[0, 0.1], [0, 0.1]])
data_from_this_model_lin = etomo.buffer_correction(data_from_this_model_lin,
                                                   coords=[[0, 0.1], [0, 0.1]])

#%%

# Compare PSF's of TEM-Simulator and this implementation

dir_path_ball = os.path.dirname('/home/zickert/odl/odl/contrib/etomo/examples/test_forward_op/test_forward_op.py')
file_path_ball = os.path.join(dir_path, 'test.mrc')


space_3d = odl.uniform_discr(min_pt=[-10]*3, max_pt=[10]*3,
                             shape=(100, 100, 100),
                             dtype='float32')
sample_height = 20e-9
space_3d = reco_space
radius = 5
value = 100/(sigma*sample_height)
cylinder = space_3d.element(lambda x: sum(xi**2 for xi in x[:2]) < radius ** 2)
cylinder *= value


probe = -100/(sigma*sample_height)*reco_space.one()
probe += cylinder

# Create a minimal header. All parameters except these here have default
# values.
#header = mrc_header_from_params(ball.shape, ball.dtype, kind='volume')


#with FileWriterMRC(file_path_ball, header) as writer:
#    # Write both header and data to the file
#    writer.write(ball.asarray())
#
#with FileReaderMRC(file_path_ball) as reader:
#    # Get header and data
#    header, data = reader.read()

#plt.figure()
#plt.imshow(data[:,:,20], cmap='Greys_r', interpolation='none')
#plt.show()

cylinder.show(coords = [None,None, -10])
cylinder_data = forward_op(probe)
cylinder_data.show(coords = [0,None,None])