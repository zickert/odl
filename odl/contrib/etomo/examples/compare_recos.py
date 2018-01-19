import os
from odl.contrib.mrc import FileReaderMRC
import odl
import numpy as np

dir_path = os.path.abspath('/home/zickert/TEM_reco_project/Reconstructions/from_NAS/')
file_path_reco = os.path.join(dir_path, 'rec_T06_800.mrc')


with FileReaderMRC(file_path_reco) as reco_reader:
    reco_data = reco_reader.read_data(swap_axes=False)
    reco_header = reco_reader.read_header()

M = 25000
rescale_factor = 1e9
    
# Reconstruction space: discretized functions on a cuboid
reco_space = odl.uniform_discr(min_pt=[-rescale_factor*210e-9/4,
                                       -rescale_factor*250e-9/4,
                                       -rescale_factor*40e-9/4],
                               max_pt=[rescale_factor*210e-9/4,
                                       rescale_factor*250e-9/4,
                                       rescale_factor*40e-9/4],
                               shape=[210, 250, 40], dtype='float64')


reco = reco_space.element(reco_data)

