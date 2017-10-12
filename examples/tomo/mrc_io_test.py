import os
import odl
import numpy as np
from odl.contrib.mrc import (FileReaderMRC)
import matplotlib.pyplot as plt

dir_path = os.path.abspath('/home/zickert/One_particle/')
file_path_phantom = os.path.join(dir_path, 'rna_phantom.mrc')
file_path_re_map = os.path.join(dir_path, '1I3Q_map.mrc')
file_path_im_map = os.path.join(dir_path, '1I3Q_abs_map.mrc')
file_path_tiltseries = os.path.join(dir_path, 'tiltseries.mrc')
file_path_tiltseries_nonoise = os.path.join(dir_path, 'tiltseries_nonoise.mrc')

with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
with FileReaderMRC(file_path_tiltseries) as tiltseries_reader:
    tiltseries_header, data_asarry = tiltseries_reader.read()
with FileReaderMRC(file_path_tiltseries_nonoise) as tiltseries_nonoise_reader:
    tiltseries_nonoise_header, data_nonoise_asarray = tiltseries_nonoise_reader.read()
with FileReaderMRC(file_path_re_map) as re_map_reader:
    re_map_header, re_map_asarray = re_map_reader.read()
with FileReaderMRC(file_path_im_map) as im_map_reader:
    im_map_header, im_map_asarray = im_map_reader.read()

reco_space = odl.uniform_discr(min_pt=[-10] * 3, max_pt=[10] * 3,
                               shape=[95, 100, 80], dtype='float32')
reco_space_map = odl.uniform_discr(min_pt=[-10] * 3, max_pt=[10] * 3,
                                   shape=[142, 153, 157], dtype='complex64')

angle_partition = odl.uniform_partition(-np.pi/3, np.pi/3, 61)
detector_partition = odl.uniform_partition([-10] * 2, [10] * 2, [200] * 2)
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
phantom = reco_space.element(phantom_asarray)

# subtract background
#phantom = phantom -4.877

data = ray_trafo(phantom)

ray_trafo_map = odl.tomo.RayTransform(reco_space_map, geometry)
re_map = reco_space_map.element(re_map_asarray)
im_map = reco_space_map.element(im_map_asarray)
complex_map = re_map + 1j * im_map

# subtract background
#complex_map = complex_map - 4.877 - 0.824j

data_map = ray_trafo_map(complex_map)




phantom.show(coords = [None,0,None])

data.show(coords = [0,None,None])
complex_map.show(coords = [None,0,None])


data_map.show(coords = [0,None,None])

data_nonoise_asarray_slice = data_nonoise_asarray[:,:,0]
plt.imshow(data_nonoise_asarray_slice,cmap='gray')

tr = np.transpose(data_nonoise_asarray)
tr_data = ray_trafo_map.range.element(tr)

tr_slice = np.array(tr_data)[0,:,:]

tr_slice-data_nonoise_asarray_slice

