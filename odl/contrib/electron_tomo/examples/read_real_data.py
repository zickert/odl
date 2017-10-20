#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:43:49 2017

@author: zickert
"""

import os
from odl.contrib.mrc import (FileReaderMRC)
import matplotlib.pyplot as plt

dir_path = os.path.abspath('/home/zickert/TEM_reco_project')
file_path_data = os.path.join(dir_path, 'region2.mrc')


with FileReaderMRC(file_path_data) as reader:
    header, data = reader.read()

sliced_data = data[:, :, 40]

plt.figure(); plt.imshow(sliced_data); plt.colorbar() #, vmin=-32768, vmax=-32500)

#plt.figure(); plt.imshow(np.mean(data, axis=2)); plt.colorbar()