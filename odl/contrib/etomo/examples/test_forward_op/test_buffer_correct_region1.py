import odl
import numpy as np
from odl.contrib import etomo
from odl.contrib import fom
from odl.contrib.mrc import FileReaderMRC
import matplotlib.pyplot as plt

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

detector_zero_level = np.min(data)

# Define properties of the optical system

M = 29370.0
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
                                                chromatic_abe=chromatic_abe,
                                                normalize=True)


# Define forward operator as a composition
forward_op = imageFormation_op * ray_trafo



base_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/'


gamma_H1 = 0.95
reg_param = 3e3
Niter_CG = 30
iterate = 242

method_path = 'Experimental/Region1/kaczmarz'
param_path = '/gamma_H1='+str(gamma_H1)+'_reg_par='+str(reg_param)+'_niter_CG='+str(Niter_CG)+'_num_cycles='+str(num_cycles)+'_iterate_' + str(iterate) 
path = base_path + method_path + param_path + '.npy'

reco_array = np.load(path)         
reco = reco_space.element(reco_array)

data = forward_op(reco)

#%%

# Test that buffer_correct method really uses background for balls phantom


dim_t, dim_x, dim_y = data.shape
data_asarray = data.asarray()

bg_coords = [[0, 0.25],[0, 0.25]]


# Pick out background according to coords
bg = data_asarray[:, round(dim_x*bg_coords[0][0]):round(dim_x*bg_coords[0][1]),
                  round(dim_y*bg_coords[1][0]):round(dim_y*bg_coords[1][1])]



plt.imshow(bg[80,:,:])
plt.colorbar()

plt.figure()
plt.imshow(data_asarray[80,:,:])
plt.colorbar()

