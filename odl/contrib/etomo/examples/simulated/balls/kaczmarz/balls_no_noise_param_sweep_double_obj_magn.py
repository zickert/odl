"""Electron tomography reconstruction example using data from TEM-Simulator"""


import numpy as np
import os
import odl
from odl.contrib import etomo
from odl.contrib.mrc import FileReaderMRC

# Read phantom and data.
dir_path = os.path.abspath('/mnt/imagingnas/data/Users/gzickert/TEM/Data/Simulated/Balls/No_noise')
file_path_phantom = os.path.join(dir_path, 'balls_phantom.mrc')
file_path_tiltseries = os.path.join(dir_path, 'tiltseries_perfect_mtf_perfect_dqe.mrc')

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

wave_length = 0.00251e-9  # m
wave_number = 2 * np.pi / wave_length

sigma = e_mass * e_charge / (wave_number * planck_bar ** 2)

abs_phase_ratio = 0.1
obj_magnitude = 2.0 * sigma / rescale_factor
num_angles = 61


num_angles_per_block = 1


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

## Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*(210e-9)/4,
                                       -rescale_factor*(250e-9)/4,
                                       -rescale_factor*(40e-9)/4],
                               max_pt=[rescale_factor*(210e-9)/4,
                                       rescale_factor*(250e-9)/4,
                                       rescale_factor*(40e-9)/4],
                               shape=[210, 250, 40], dtype='float64')
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

phantom = reco_space.element(phantom_asarray)

bg_cst = np.min(phantom)
phantom -= bg_cst


data=forward_op(phantom)

# %% RECONSTRUCTION

reg_param_list = [1e-5, 1e-4, 1e-3]
gamma_H1 = 0.0
num_angles_per_block = 1
num_cycles_list = [1,2,3,5,10]
Niter_CG = 30


reco_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Reconstructions/Simulated/Balls/no_noise_double_obj_magn/kaczmarz'


for reg_param in reg_param_list:
    for num_cycles in num_cycles_list:
        saveto_path = reco_path+'/gamma_H1='+str(gamma_H1)+'_reg_par='+str(reg_param)+'_niter_CG='+str(Niter_CG)+'_num_cycles='+str(num_cycles)+'_iterate_{}'
        
        
        
        reco = ray_trafo.domain.zero()
        callback = (odl.solvers.CallbackPrintIteration() &
                    odl.solvers.CallbackShow() &
                    odl.solvers.CallbackSaveToDisk(saveto=saveto_path,
                                                              step=num_angles*num_cycles-1,
                                                              impl='numpy'))
        
        
        
        
        kaczmarz_plan = etomo.make_kaczmarz_plan(num_angles,
                                                 block_length=num_angles_per_block,
                                                 method='mls')
        
        ray_trafo_block = ray_trafo.get_sub_operator(kaczmarz_plan[0])
        
        F_post = etomo.make_imageFormationOp(ray_trafo_block.range,
                                             wave_number, spherical_abe,
                                             defocus,
                                             rescale_factor=rescale_factor,
                                             obj_magnitude=obj_magnitude,
                                             abs_phase_ratio=abs_phase_ratio,
                                             aper_rad=aper_rad, aper_angle=aper_angle,
                                             focal_length=focal_length,
                                             mean_energy_spread=mean_energy_spread,
                                             acc_voltage=acc_voltage,
                                             chromatic_abe=chromatic_abe,
                                             normalize=True)
        F_pre = odl.IdentityOperator(reco_space)
        
        
        get_op = etomo.make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre,
                                      Op_post=F_post)
        get_data = etomo.make_data_blocks(data, kaczmarz_plan)
        
        # Optional nonnegativity-constraint
        nonneg_projection = etomo.get_nonnegativity_projection(reco_space)
        
        
        reco = reco_space.zero()
        #reco = 0.9 * phantom
        
        get_proj_op = etomo.make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre,
                                           Op_post=None)
        
        etomo.kaczmarz_SART_method(get_proj_op, reco, get_data, len(kaczmarz_plan),
                                   reg_param*obj_magnitude ** 2,
                                   imageFormationOp=F_post, gamma_H1=gamma_H1,
                                   niter_CG=Niter_CG, callback=callback,
                                   num_cycles=num_cycles, projection=nonneg_projection)
        
        

