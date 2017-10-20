"""Phase contrast TEM reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


import numpy as np
import matplotlib.pyplot as plt
import pytest
import odl
from odl.contrib.electron_tomo.constant_phase_abs_ratio import ConstantPhaseAbsRatio
from odl.contrib.electron_tomo.block_ray_trafo import BlockRayTransform
from odl.contrib.electron_tomo.kaczmarz_alg import *
from odl.contrib.electron_tomo.image_formation_etomo import *
from odl.contrib.electron_tomo.kaczmarz_util import *
from odl.contrib.electron_tomo.support_constraint import spherical_mask


obj_magnitude = 1e-2

noise_lvl = 1e-2
regpar = 1e1
num_angles = 360
num_angles_per_kaczmarz_block = 1
num_cycles = 1

det_size = 16e-6  # m
wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length
M = 25000.0
aper_rad = 0.5*40e-6  # m
focal_length = 2.7e-3  # m
spherical_abe = 2.1e-3  # m
defocus = 3e-6  # m

ctf_scaling_factor = (30 / (det_size / M * 100))

reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300])

angle_partition = odl.uniform_partition(0, np.pi, num_angles)
detector_partition = odl.uniform_partition(-30, 30, 512)

geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray_trafo = BlockRayTransform(reco_space, geometry)  # odl.tomo.RayTransform(reco_space.complex_space, geometry)

imageFormation_op = make_imageFormationOp(ray_trafo.range, 
                                          wave_number, spherical_abe, defocus,
                                          det_size, M, rescale_ctf_factor = ctf_scaling_factor,
                                          obj_magnitude=obj_magnitude)

mask = reco_space.element(spherical_mask, radius=19)

# Leave out detector operator for simplicity
forward_op = imageFormation_op * ray_trafo * mask
forward_op_linearized = forward_op.derivative(reco_space.zero())

phantom = odl.phantom.shepp_logan(reco_space, modified=True)  # (1+1j) *

data = forward_op(phantom)
noise = odl.phantom.white_noise(data.space)
data += (noise_lvl * (data.space.one()-data).norm() / noise.norm()) * noise


# %%
reco = reco_space.zero()
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())


kaczmarz_plan = make_kaczmarz_plan(num_angles,
                                   num_blocks_per_superblock = num_angles_per_kaczmarz_block, method = 'random')

ray_trafo_block = ray_trafo.get_sub_operator(kaczmarz_plan[0])


F_post = make_imageFormationOp(ray_trafo_block.range, 
                               wave_number, spherical_abe, defocus, det_size,
                               M, obj_magnitude=obj_magnitude, rescale_ctf_factor = ctf_scaling_factor)
F_pre = odl.MultiplyOperator(mask,reco_space,reco_space)

get_op = make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre, Op_post=F_post)
get_data = make_data_blocks(data, kaczmarz_plan)


# Optional nonnegativity-constraint
nonneg_constraint = odl.solvers.IndicatorNonnegativity(reco_space).proximal(1)


def nonneg_projection(x):
    x[:] = nonneg_constraint(x)


# %%
#    
#reco = reco_space.zero()
#kaczmarz_reco_method(get_op, reco, get_data, len(kaczmarz_plan),
#                     regpar*obj_magnitude ** 2, callback=callback,
#                     num_cycles=num_cycles, projection=nonneg_projection)
#

# %%

reco = reco_space.zero()
get_proj_op = make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre, Op_post=None)

kaczmarz_SART_method(get_proj_op, reco, get_data, len(kaczmarz_plan),
                     regpar*obj_magnitude ** 2, imageFormationOp = F_post,
                     callback=callback, num_cycles=num_cycles, projection=nonneg_projection)

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

