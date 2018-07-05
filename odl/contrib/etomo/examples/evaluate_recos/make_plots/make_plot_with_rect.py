import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
# Make plots paper
import os
from odl.contrib.mrc import FileReaderMRC
import numpy as np
import mrcfile
import odl
from odl.contrib import etomo
from skimage import exposure

from matplotlib_scalebar.scalebar import ScaleBar


NAS_data_path = '/mnt/imagingnas/data/Users/gzickert/TEM'
NAS_reco_path = NAS_data_path + '/Reconstructions'
NAS_plots_path = NAS_data_path + '/Plots'
# Plots of old reconstructions



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

#if not os.path.isdir(os.path.dirname(fig_path)):
#    os.makedirs(os.path.dirname(fig_path))
#
#plt.savefig(fig_path, bbox_inches='tight', dmi = 1000)

    
#%%


    
im = orthoslice

# Create figure and axes
fig,ax = plt.subplots(1)

ax.axis('off')

# Display the image
ax.imshow(im, origin='lower', cmap='bone')

## Create a Rectangle patch
rect1 = patches.Rectangle((2048-256,2200-128),512,256,linewidth=1,edgecolor='r',facecolor='none')
rect2 = patches.Rectangle((1530-256,1320-128),512,256,linewidth=1,edgecolor='r',facecolor='none')
rect3 = patches.Rectangle((3481-256,2525-128),512,256,linewidth=1,edgecolor='r',facecolor='none')

#rect1 = patches.Rectangle((2200-256,2048-128),512,256,linewidth=1,edgecolor='r',facecolor='none')
#rect2 = patches.Rectangle((1320-256,1530-128),512,256,linewidth=1,edgecolor='r',facecolor='none')
#rect3 = patches.Rectangle((2525-256,3481-128),512,256,linewidth=1,edgecolor='r',facecolor='none')
#

# Add the patch to the Axes
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)


# Add scale-bar   
M = 29370.0
det_pix_size = 14e-6
scalebar = ScaleBar(det_pix_size/M, length_fraction=0.3)
plt.gca().add_artist(scalebar)


plt.show()

fig_path = NAS_plots_path + '/Data/Experimental/' + path[:-4] + '_sub_regs_marked.png'


fig.savefig(fig_path, bbox_inches='tight', dpi = 1000)

#%%
