"""Cone beam 2D example for checking that orientations are handled correctly.

Due to differing axis conventions between ODL and the ray transform
back-ends, a check is needed to confirm that the translation steps are
done correctly.

Both pairs of plots of ODL projections and NumPy axis sums should look
similar in the sense that they should show the same features in the
right arrangement (not flipped, rotated, etc.), up to differences in
magnification and other cone effects.

This example is best run in Spyder section-by-section (CTRL-Enter).
"""

# %% Set up the things that never change

import matplotlib.pyplot as plt
import numpy as np
import odl

# Set back-end here (for `None` the fastest available is chosen)
impl = None
# Set a volume shift. This should move the projections in the same direction,
# and shifts along the projection axis should only change cone effects.
shift = np.array([0.0, 25.0])

img_shape = (100, 150)
img_max_pt = np.array(img_shape, dtype=float) / 2
img_min_pt = -img_max_pt
reco_space = odl.uniform_discr(img_min_pt + shift, img_max_pt + shift,
                               img_shape, dtype='float32')
phantom = odl.phantom.indicate_proj_axis(reco_space)

assert np.allclose(reco_space.cell_sides, 1)

# Check projections at 0, 90, 180 and 270 degrees
grid = odl.RectGrid([0, np.pi / 2, np.pi, 3 * np.pi / 2])
angle_partition = odl.uniform_partition_fromgrid(grid)

# Make detector large enough to cover the object
src_radius = 500
det_radius = 1000
fan_angle = np.arctan(img_max_pt[1] / src_radius)
det_size = np.floor(2 * (src_radius + det_radius) * np.sin(fan_angle))
det_shape = int(det_size)
det_max_pt = det_size / 2
det_min_pt = -det_max_pt
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, det_shape)

assert np.allclose(detector_partition.cell_sides, 1)

# Sum manually using Numpy
sum_along_x = np.sum(phantom, axis=0)
sum_along_y = np.sum(phantom, axis=1)

geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                    src_radius, det_radius)
# Check initial configuration
assert np.allclose(geometry.det_axis_init, [1, 0])
assert np.allclose(geometry.src_to_det_init, [0, 1])

# Create projections
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl=impl)
proj_data = ray_trafo(phantom)


# %% Test forward projection along y axis


# Axis in this image is x. This corresponds to 0 degrees.
proj_data.show(indices=[0, None],
               title='Projection at 0 degrees ~ Sum along y axis')
fig, ax = plt.subplots()
ax.plot(sum_along_y)
ax.set_xlabel('x')
plt.title('Sum along y axis')
plt.show()
# Check axes in geometry
axis_sum_y = geometry.det_axis(np.deg2rad(0))
assert np.allclose(axis_sum_y, [1, 0])


# %% Test forward projection along x axis


# Axis in this image is y. This corresponds to 90 degrees.
proj_data.show(indices=[1, None],
               title='Projection at 90 degrees ~ Sum along x axis')
fig, ax = plt.subplots()
ax.plot(sum_along_x)
ax.set_xlabel('y')
plt.title('Sum along x axis')
plt.show()
# Check axes in geometry
axis_sum_x = geometry.det_axis(np.deg2rad(90))
assert np.allclose(axis_sum_x, [0, 1])
