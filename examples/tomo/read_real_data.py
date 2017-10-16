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
file_path_data = os.path.join(dir_path, 'rk7.mrc')


with FileReaderMRC(file_path_data) as reader:
    header, data = reader.read()

sliced_data = data[600:700, 900:1000, 30]

plt.imshow(sliced_data, vmin=-32768, vmax=-32500)
