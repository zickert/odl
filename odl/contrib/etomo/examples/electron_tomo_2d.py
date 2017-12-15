"""Electron tomography reconstruction example in 2d."""


import numpy as np
import matplotlib.pyplot as plt
import odl
from odl.contrib import etomo


obj_magnitude = 1e-2

# Relative noise level
noise_lvl = 1e-2

regpar = 1e1

num_angles = 360
num_angles_per_block = 1
num_cycles = 1

wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length

aper_rad = 0.5*40e-6  # m
focal_length = 2.7e-3  # m
spherical_abe = 2.1e-3  # m
defocus = 3e-6  # m

# In this toy example, rescale_factor can be chosen arbitrarily, but 0.5e9 will
# give a particle of roughly the same size as the rna_phantom from the
# TEM-Simulator. A greater value of this factor means that the corresponding
# 'true particle' is smaller, hence varies on a larger scale in frequency-space
# This in turn means that the CTF will have a greater effect, leading to more
# 'fringes' in the data.
rescale_factor = 0.5e9

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(min_pt=[-20] * 2, max_pt=[20] * 2,
                               shape=[300] * 2)

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = num_angles, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, num_angles)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform
ray_trafo = etomo.BlockRayTransform(reco_space, geometry)

# The image-formation operator models the optics and the detector
# of the electron microscope.
imageFormation_op = etomo.make_imageFormationOp(ray_trafo.range,
                                                wave_number, spherical_abe,
                                                defocus,
                                                rescale_factor=rescale_factor,
                                                obj_magnitude=obj_magnitude)

# Define a spherical mask to implement support constraint.
mask = reco_space.element(etomo.spherical_mask, radius=19)

# Define forward operator as a composition
forward_op = imageFormation_op * ray_trafo * mask
# forward_op_linearized = forward_op.derivative(reco_space.zero())

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)  # (1+1j) *

# Create data by calling the forward operator on the phantom
data = forward_op(phantom)

# Add noise to the data
noise = odl.phantom.white_noise(data.space)
data += (noise_lvl * (data.space.one()-data).norm() / noise.norm()) * noise

# Compare with affine approximation
data.show()
aff_apprx = forward_op(reco_space.zero()) + forward_op.derivative(reco_space.zero())
aff_apprx_data = aff_apprx(phantom)
aff_apprx_data.show()
non_linearity = data - aff_apprx_data
non_linearity.show()
forward_op(reco_space.zero()).show()

# %%

# Choose a starting point
reco = reco_space.zero()

# Optional: pass callback objects to solver
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

kaczmarz_plan = etomo.make_kaczmarz_plan(num_angles,
                                         block_length=num_angles_per_block,
                                         method='mls')

ray_trafo_block = ray_trafo.get_sub_operator(kaczmarz_plan[0])


F_post = etomo.make_imageFormationOp(ray_trafo_block.range,
                                     wave_number, spherical_abe, defocus,
                                     obj_magnitude=obj_magnitude,
                                     rescale_factor=rescale_factor)
F_pre = odl.MultiplyOperator(mask, reco_space, reco_space)

get_op = etomo.make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre,
                              Op_post=F_post)
get_data = etomo.make_data_blocks(data, kaczmarz_plan)

# Optional nonnegativity-constraint
nonneg_projection = etomo.get_nonnegativity_projection(reco_space)


# %%

# Reset starting point
reco = reco_space.zero()
get_proj_op = etomo.make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre,
                                   Op_post=None)

etomo.kaczmarz_SART_method(get_proj_op, reco, get_data, len(kaczmarz_plan),
                           regpar*obj_magnitude ** 2,
                           imageFormationOp=F_post, callback=callback,
                           num_cycles=num_cycles, projection=nonneg_projection)

# %%

#reco = reco_space.zero()
#etomo.kaczmarz_reco_method(get_op, reco, get_data, len(kaczmarz_plan),
#                     regpar*obj_magnitude ** 2, callback=callback,
#                     num_cycles=num_cycles, projection=nonneg_projection)
#



#
#odl.solvers.conjugate_gradient_normal(forward_op.derivative(reco), reco, data - forward_op(reco),
#                                      niter=100, callback=callback)

# non-linear cg must be adapted to complex case
#func = odl.solvers.L2NormSquared(data.space).translated(data) * forward_op
#odl.solvers.conjugate_gradient_nonlinear(func, reco, callback=callback)#, line_search=1e0, callback=callback,
#                                         #nreset=50)
#                                        
# Landweber iterations
#odl.solvers.landweber(forward_op, reco, data, 1000, omega=3e1, callback=callback)


#if __name__ == '__main__':
#    odl.util.test_file(__file__)
#
#
#    x_adj = reco_space.one() + odl.phantom.white_noise(reco_space)
#    y_adj = forward_op_linearized.range.one() + odl.phantom.white_noise(forward_op_linearized.range)
#
#    Ax_adj = forward_op_linearized(x_adj)
#    ATy_adj = forward_op_linearized.adjoint(y_adj)
#
#    ip1 = x_adj.inner(ATy_adj)
#    ip2 = Ax_adj.inner(y_adj)
#
#    assert pytest.approx(ip1.real,rel=5e-2) == ip2.real
