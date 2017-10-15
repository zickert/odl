import os
import odl
import numpy as np
from odl.contrib.mrc import (FileReaderMRC)
import matplotlib.pyplot as plt

dir_path = os.path.abspath('/home/zickert/TEM_reco_project/One_particle_new_simulation')
file_path_phantom = os.path.join(dir_path, 'rna_phantom.mrc')
file_path_phantom_abs = os.path.join(dir_path, 'rna_phantom_abs.mrc')
file_path_tiltseries = os.path.join(dir_path, 'tiltseries.mrc')
file_path_tiltseries_nonoise = os.path.join(dir_path, 'tiltseries_nonoise.mrc')

with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
with FileReaderMRC(file_path_phantom_abs) as phantom_abs_reader:
    phantom_abs_header, phantom_abs_asarray = phantom_abs_reader.read()
with FileReaderMRC(file_path_tiltseries) as tiltseries_reader:
    tiltseries_header, data_asarry = tiltseries_reader.read()
with FileReaderMRC(file_path_tiltseries_nonoise) as tiltseries_nonoise_reader:
    tiltseries_nonoise_header, data_nonoise_asarray = tiltseries_nonoise_reader.read()

reco_space = odl.uniform_discr(min_pt=[-30] * 3, max_pt=[30] * 3,
                               shape=[95, 100, 80], dtype='complex128')

angle_partition = odl.uniform_partition(-np.pi/3, np.pi/3, 61)

# Make detector smaller than reco_space, so that we dont see edge of background
detector_partition = odl.uniform_partition([-30] * 2, [30] * 2, [200] * 2)

# The x-axis is the tilt-axis.
# Check that the geometry matches the one from TEM-simulator!
# In particular, check that det_pos_init and det_axes_init are correct.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=(1, 0, 0),
                                           det_pos_init=(0, 0, -1),
                                           det_axes_init=((1, 0, 0), (0, 1, 0)))
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

phantom = reco_space.element(phantom_asarray + 1j * phantom_abs_asarray)
# subtract background
#phantom = phantom - 4.877 - 0.824j


data = ray_trafo(phantom)

phantom.show(coords=[None, None,0])

data.show(coords=[0, None, None])

# The last index in MRC-file corresponds to tilt-angle
data_nonoise_asarray_slice = data_nonoise_asarray[:, :, 30]
plt.imshow(data_nonoise_asarray_slice, cmap='gray')
plt.colorbar()
