from time import sleep
import matplotlib.pyplot as plt
import numpy as np

__all__ = ('plot_3d_ortho_slices', 'plot_3d_axis_drive')


def plot_3d_ortho_slices(obj_3d):

    centers = 0.5*(obj_3d.space.min_pt + obj_3d.space.max_pt)
    obj_3d.show(coords=[centers[0], None, None])
    obj_3d.show(coords=[None, centers[1], None])
    obj_3d.show(coords=[None, None, centers[2]])


def plot_3d_axis_drive(obj_3d, axis=0, dt=0.3, fix_color_scale=True):
    fig = plt.figure()
    if fix_color_scale:
        cmin = np.min(obj_3d.asarray())
        cmax = np.max(obj_3d.asarray())
    for slice_idx in range(obj_3d.shape[axis]):
        if axis == 0:
            indices = [slice_idx, None, None]
        elif axis == 1:
            indices = [None, slice_idx, None]
        else:
            indices = [None, None, slice_idx]
        if fix_color_scale:
            obj_3d.show(fig=fig, indices=indices, clim=(cmin, cmax))
        else:
            obj_3d.show(fig=fig, indices=indices)
        sleep(dt)
