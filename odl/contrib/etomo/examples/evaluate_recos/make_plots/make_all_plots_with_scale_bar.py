# Make plots paper
import os
from odl.contrib.mrc import FileReaderMRC
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import odl
from odl.contrib import etomo
from skimage import exposure

from matplotlib_scalebar.scalebar import ScaleBar



NAS_data_path = '/mnt/imagingnas/data/Users/gzickert/TEM'
NAS_reco_path = NAS_data_path + '/Reconstructions'
#NAS_plots_path = NAS_data_path + '/Plots'
# Use Dropbox folder instead
NAS_plots_path = '/home/zickert/Dropbox/ET-paper/Plots'


# Plots of old reconstructions
#%%
NAS_ref_path = '/mnt/imagingnas/Reference/TEM/Reconstructions/2013_Handbook_of_Mathematical_Methods_in_Imaging'
path_list = []
# Approximate inverse, region 1-3
path_list.append('/Etreco/FEI_Region_1/FEI_region1_rec_c_2_g_42_rotated.mrc')
path_list.append('/Etreco/FEI_Region_2/FEI_region2_rec_c_2_g_42_rotated.mrc')
path_list.append('/Etreco/FEI_Region_3/FEI_region3_rec_c_2_g_42_rotated.mrc')
# TVreg, region 1-3
path_list.append('/TVreg/FEI_Region_1/TVreg_region1_p100_1500.mrc')
path_list.append('/TVreg/FEI_Region_2/TVreg_region2_p100_1500.mrc')
path_list.append('/TVreg/FEI_Region_3/TVreg_region3_p100_1500.mrc')
# Sirt, region 1-3
path_list.append('/IMOD WBP & SIRT/FEI_Region_1/reconstructions/FEI_region1_IMOD_SIRT_rec_reg010_cutoff005_iter10.mrc')
path_list.append('/IMOD WBP & SIRT/FEI_Region_2/reconstructions/FEI_region2_IMOD_SIRT_rec_reg010_cutoff005_iter10.mrc')
path_list.append('/IMOD WBP & SIRT/FEI_Region_3/reconstructions/FEI_region3_IMOD_SIRT_rec_reg010_cutoff005_iter10.mrc')
# WBP, regions 1-3
path_list.append('/IMOD WBP & SIRT/FEI_Region_1/reconstructions/FEI_region1_IMOD_WBP_rec_reg110_cutoff020.mrc')
path_list.append('/IMOD WBP & SIRT/FEI_Region_2/reconstructions/FEI_region2_IMOD_WBP_rec_reg120_cutoff040.mrc')
path_list.append('/IMOD WBP & SIRT/FEI_Region_3/reconstructions/FEI_region3_IMOD_WBP_rec_reg101_cutoff050.mrc')

for path in path_list:
    if 'Region_1' in path:
        z_coord = 99
    elif 'Region_2' in path:
        z_coord = 174
    else:
        z_coord = 290
    if 'TVreg' in path:
        ## comment out line 619 in mrc.py in order to be able to read:
        with FileReaderMRC(NAS_ref_path + path) as reco_reader:
            header, reco = reco_reader.read()
    else:
        with mrcfile.open(NAS_ref_path + path) as reco_reader:
            reco = reco_reader.data
    reco = np.transpose(reco, axes=(2, 1, 0))   
    plt.figure()         
    plt.axis('off')
    orthoslice = reco[:, :, z_coord].T
    orthoslice = exposure.equalize_hist(orthoslice)   
    plt.imshow(orthoslice, origin='lower', cmap='bone')

    # Add scale-bar   
    M = 29370.0
    det_pix_size = 14e-6
    scalebar = ScaleBar(det_pix_size/M, length_fraction=0.3)
    plt.gca().add_artist(scalebar)
    plt.show()

    fig_path = NAS_plots_path + '/Reconstructions/Experimental/bookchapter' + path[:-4] + '_hist_eq' + '.png'

    if not os.path.isdir(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))

    plt.savefig(fig_path, bbox_inches='tight')
#%%
# Plots of kaczmarz recos, region 1-3
path_list = []
path_list.append('/Experimental/Region1/kaczmarz/gamma_H1=0.95_reg_par=3000.0_niter_CG=30_num_cycles=3_iterate_242.npy')
path_list.append('/Experimental/Region2/kaczmarz/gamma_H1=0.95_reg_par=3000.0_niter_CG=30_num_cycles=3_iterate_242.npy')
path_list.append('/Experimental/Region3/kaczmarz/gamma_H1=0.95_reg_par=3000.0_niter_CG=30_num_cycles=3_iterate_242.npy')
 
for path in path_list:
    if 'Region1' in path:
        z_coord = 99
    elif 'Region2' in path:
        z_coord = 174
    else:
        z_coord = 290

    reco = np.load(NAS_reco_path + path)
    plt.figure()         
    plt.axis('off')
    orthoslice = reco[:, :, z_coord].T
    orthoslice = exposure.equalize_hist(orthoslice)   
    plt.imshow(orthoslice, origin='lower', cmap='bone')

    # Add scale-bar   
    M = 29370.0
    det_pix_size = 14e-6
    scalebar = ScaleBar(det_pix_size/M, length_fraction=0.3)
    plt.gca().add_artist(scalebar)
    plt.show()


    fig_path = NAS_plots_path + '/Reconstructions' + path[:-4] + '_hist_eq' + '.png'

    if not os.path.isdir(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))

    plt.savefig(fig_path, bbox_inches='tight')
    
#%%
# Plot experimental data 
dir_path = '/mnt/imagingnas/Reference/TEM/Data/Tilt-series/Experimental/FEI/In-vitro_test_specimen'
path_list = ['/region1.mrc', '/region2.mrc', '/region3.mrc']



for path in path_list:
    with FileReaderMRC(dir_path + path) as reader:
        header, data = reader.read()
    data = np.transpose(data, axes=(2, 0, 1))   
    plt.figure()         
    plt.axis('off')
    orthoslice = data[40,:, :].T
    plt.imshow(orthoslice, origin='lower', cmap='bone')

    # Add scale-bar   
    M = 29370.0
    det_pix_size = 14e-6
    scalebar = ScaleBar(det_pix_size/M, length_fraction=0.3)
    plt.gca().add_artist(scalebar)
    plt.show()
    
    fig_path = NAS_plots_path + '/Data/Experimental/' + path[:-4] + '.png'
    
    if not os.path.isdir(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    
    plt.savefig(fig_path, bbox_inches='tight')
    
#%%
# Plot large micrograph containing experimental sub-data
dir_path = '/mnt/imagingnas/Reference/TEM/Data/Tilt-series/Experimental/FEI/In-vitro_test_specimen'
path = '/series2_aligned.mrc'


with FileReaderMRC(dir_path + path) as reader:
    header, data = reader.read()
data = np.transpose(data, axes=(2, 0, 1))   
plt.figure()         
plt.axis('off')
orthoslice = data[40,:, :].T
plt.imshow(orthoslice, origin='lower', cmap='bone')

fig_path = NAS_plots_path + '/Data/Experimental/' + path[:-4] + '.png'

if not os.path.isdir(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))

plt.savefig(fig_path, bbox_inches='tight', dpi = 1000)

    



    
#%%    
# Plots recos for the balls
path_list = []
path_list.append('/Simulated/Balls/dose_6000/kaczmarz/_gamma_H1=0.95_reg_par=400.0_niter_CG=30_num_cycles=3_iterate_182.npy')
path_list.append('/Simulated/Balls/dose_6000/landweber/omega=155.08468399169368_iterate_50.npy')
path_list.append('/Simulated/Balls/dose_6000/pdhg_tv_pos_constr/step_par=0.1_reg_par=0.0003_iterate_3000.npy')
path_list.append('/Simulated/Balls/dose_6000/pdhg_tv_pos_constr/step_par=0.1_reg_par=0.0003_iterate_10000.npy')
path_list.append('/Simulated/Balls/dose_6000/gradient_descent_huber_reg/_gamma=0.01_reg_par=0.000375_iterate_3000.npy')
path_list.append('/Simulated/Balls/dose_6000/gradient_descent_huber_reg/_gamma=0.01_reg_par=0.000375_iterate_10000.npy')

path_list.append('/Simulated/Balls/no_noise/kaczmarz/gamma_H1=0.0_reg_par=0.001_niter_CG=30_num_cycles=1_iterate_60.npy')
path_list.append('/Simulated/Balls/no_noise_double_obj_magn/kaczmarz/gamma_H1=0.0_reg_par=0.001_niter_CG=30_num_cycles=1_iterate_60.npy')
path_list.append('/Simulated/Balls/no_noise/landweber/omega=155.08468399169368_iterate_10000.npy')
path_list.append('/Simulated/Balls/no_noise_double_obj_magn/landweber/omega=46.72091587945816_iterate_10000.npy')
path_list.append('/Simulated/Balls/no_noise/landweber_lin/omega=155.08468399169368_iterate_10000.npy')
path_list.append('/Simulated/Balls/no_noise_double_obj_magn/landweber_lin/omega=46.72091587945816_iterate_10000.npy')



for path in path_list:
    nz = 40
    reco = np.load(NAS_reco_path + path)
    plt.figure()         
    plt.axis('off')
    orthoslice = reco[:, :, (nz//2) -1].T
    plt.imshow(orthoslice, origin='lower', cmap='bone')


    # Add scale-bar   
    #M = 25000.0
    vox_size = 5e-10
    scalebar = ScaleBar(vox_size, length_fraction=0.3)
    plt.gca().add_artist(scalebar)
    plt.show()



    fig_path = NAS_plots_path + '/Reconstructions' + path[:-4] + '.png'

    if not os.path.isdir(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))

    plt.savefig(fig_path, bbox_inches='tight')
    
    
#%%
#  Plot some zoom-in's of balls reco
 path_list = []
path_list.append('/Simulated/Balls/no_noise/kaczmarz/gamma_H1=0.0_reg_par=0.001_niter_CG=30_num_cycles=1_iterate_60.npy')
path_list.append('/Simulated/Balls/no_noise_double_obj_magn/landweber_lin/omega=46.72091587945816_iterate_10000.npy')



for path in path_list:
    nz = 40
    reco = np.load(NAS_reco_path + path)
    plt.figure()         
    plt.axis('off')
    orthoslice = reco[0:105, 10:95, (nz//2) -1].T
    plt.imshow(orthoslice, origin='lower', cmap='bone')


    fig_path = NAS_plots_path + '/Reconstructions' + path[:-4] + '_zoom' + '.png'

    if not os.path.isdir(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))

    plt.savefig(fig_path, bbox_inches='tight')   
    
    
#%%
# Plot balls phantom
dir_path = '/mnt/imagingnas/data/Users/gzickert/TEM/Data/Simulated/Balls/dose_6000'
path = '/balls_phantom.mrc'

with FileReaderMRC(dir_path + path) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
plt.figure()         
plt.axis('off')
orthoslice = phantom_asarray[:,:,19].T
plt.imshow(orthoslice, origin='lower', cmap='bone')


# Add scale-bar   
#M = 25000.0
vox_size = 5e-10
scalebar = ScaleBar(vox_size, length_fraction=0.3)
plt.gca().add_artist(scalebar)
plt.show()


fig_path = NAS_plots_path + '/Phantom/' + path[:-4] + '.png'

if not os.path.isdir(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))

plt.savefig(fig_path, bbox_inches='tight')
#%%
# Plot noise-free and noisy balls data

# Read phantom and data.
dir_path = os.path.abspath('/mnt/imagingnas/data/Users/gzickert/TEM/Data/Simulated/Balls/dose_6000')
file_path_phantom = os.path.join(dir_path, 'balls_phantom.mrc')
#file_path_tiltseries = os.path.join(dir_path, 'tiltseries_perfect_mtf_perfect_dqe.mrc')

with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
#with FileReaderMRC(file_path_tiltseries) as tiltseries_reader:
#    tiltseries_header, data_asarray = tiltseries_reader.read()

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
#regpar = 3e3
#gamma_H1 = 0.9
num_angles = 61
num_angles_per_block = 1

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
total_dose = 6000e18  # m^-2
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
                                                obj_magnitude=obj_magnitude,
                                                rescale_factor=rescale_factor,
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

# We remove the background from the phantom
bg_cst = np.min(phantom)
phantom -= bg_cst

# Create data by calling forward op on phantom
data = forward_op(phantom)

#### plotting
plt.figure()         
plt.axis('off')
orthoslice = data.asarray()[30,:,:].T
plt.imshow(orthoslice, origin='lower', cmap='bone')



# Add scale-bar   
M = 25000.0
det_pix_size = 16e-6
scalebar = ScaleBar(det_pix_size/M, length_fraction=0.3)
plt.gca().add_artist(scalebar)
plt.show()

fig_path = NAS_plots_path + '/Data/Simulated/balls_data_no_noise.png'

if not os.path.isdir(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))

plt.savefig(fig_path, bbox_inches='tight')
####

# Add noise
total_cst = dose_per_img
total_cst *= (1 / M) ** 2
total_cst *= det_area
buffer_contr = data.space.element(etomo.buffer_contribution,
                                  sigma=sigma,
                                  ice_thickness=ice_thickness)
data = total_cst * buffer_contr * data
data = gain * odl.phantom.poisson_noise(data)

#### plotting
plt.figure()         
plt.axis('off')
orthoslice = data.asarray()[30,:,:].T
plt.imshow(orthoslice, origin='lower', cmap='bone')

M = 25000.0
det_pix_size = 16e-6
scalebar = ScaleBar(det_pix_size/M, length_fraction=0.3)
plt.gca().add_artist(scalebar)
plt.show()


fig_path = NAS_plots_path + '/Data/Simulated/balls_dose_6000_scale_bar.png'

if not os.path.isdir(os.path.dirname(fig_path)):
    os.makedirs(os.path.dirname(fig_path))

plt.savefig(fig_path, bbox_inches='tight')
####
