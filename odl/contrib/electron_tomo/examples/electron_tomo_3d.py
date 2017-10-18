"""Phase contrast TEM reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import odl
from timeit import timeit
from odl.contrib.electron_tomo.constant_phase_abs_ratio import ConstantPhaseAbsRatio
from odl.contrib.electron_tomo.block_ray_trafo import BlockRayTransform
from odl.contrib.electron_tomo.kaczmarz_alg import *
from odl.contrib.electron_tomo.image_formation_etomo import *
from odl.contrib.electron_tomo.kaczmarz_util import *
from odl.contrib.electron_tomo.support_constraint import spherical_mask

def circular_mask(x, **kwargs):
    radius = kwargs.pop('radius')
    norm_sq = np.sum(xi ** 2 for xi in x[:])

    return norm_sq <= radius ** 2



obj_magnitude = 1e-2
noise_lvl = 1e-1
num_angles = 120
regpar = 1e-1

wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length


# Define properties of the optical system
# Set focal_length to be the focal_length of the principal (first) lens !
M = 25000.0
aper_rad = 0.5*40e-6  # m
focal_length = 2.7e-3  # m
spherical_abe = 2.1e-3  # m
defocus = 3e-6  # m


# Set size of detector pixels (before rescaling to account for magnification)
det_size = 16e-6  # m

reco_space = odl.uniform_discr(min_pt=[-20]*3,
                               max_pt=[20]*3,
                               shape=[300] * 3,)

angle_partition = odl.uniform_partition(0, np.pi, num_angles)
detector_partition = odl.uniform_partition([-30] * 2, [30] * 2, [200] * 2)

geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)
ray_trafo = BlockRayTransform(reco_space, geometry)

imageFormation_op = make_imageFormationOp(ray_trafo.range, 
                                          wave_number, spherical_abe, defocus,
                                          det_size, M,
                                          obj_magnitude=obj_magnitude)

mask = reco_space.element(spherical_mask, radius=19)

forward_op = imageFormation_op * ray_trafo  * mask

f_op_lin = forward_op.derivative(reco_space.zero())

phantom = odl.phantom.shepp_logan(reco_space, modified=True)


data = forward_op(phantom)
noise = odl.phantom.white_noise(data.space)
data += (noise_lvl * (data.space.one()-data).norm() / noise.norm()) * noise

data_lin = f_op_lin(phantom)
noise = odl.phantom.white_noise(data.space)
data_lin += (noise_lvl * (data.space.one()-data).norm() / noise.norm()) * noise


data.show()


reco = reco_space.zero()
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())


kaczmarz_plan = make_kaczmarz_plan(num_angles, num_blocks_per_superblock=3,
                                   method='random')

ray_trafo_block = ray_trafo.get_sub_operator(kaczmarz_plan[0])


F_post = make_imageFormationOp(ray_trafo_block.range,
                               wave_number, spherical_abe, defocus, det_size,
                               M, obj_magnitude=obj_magnitude)

F_pre = odl.MultiplyOperator(mask, reco_space, reco_space)

get_op = make_Op_blocks(kaczmarz_plan, ray_trafo, Op_pre=F_pre, Op_post=F_post)
get_data = make_data_blocks(data, kaczmarz_plan)

# Optional nonnegativity-constraint
nonneg_constraint = odl.solvers.IndicatorNonnegativity(reco_space).proximal(1)


def nonneg_projection(x):
    x[:] = nonneg_constraint(x)


reco = reco_space.zero()
kaczmarz_reco_method(get_op, reco, get_data, len(kaczmarz_plan),
                     regpar * obj_magnitude ** 2, callback=callback,
                     num_cycles=3, niter_CG=10, do_CG_on_data_space = False,
                     projection=nonneg_projection)


reco_lin = reco_space.zero()

odl.solvers.conjugate_gradient_normal(forward_op.derivative(reco_lin),
                                      reco_lin, data - forward_op(reco_lin),
                                      niter=100, callback=callback)


#reco = reco_space.zero()
#callback = (odl.solvers.CallbackPrintIteration() &
#            odl.solvers.CallbackShow())
#odl.solvers.conjugate_gradient_normal(forward_op, reco, data,
#                                      niter=10, callback=callback)
## non-linear cg must be adapted to complex case
##func = odl.solvers.L2NormSquared(data.space).translated(data) * forward_op
##odl.solvers.conjugate_gradient_nonlinear(func, reco, line_search=1e0, callback=callback,
##                                         nreset=50)
#
##func = odl.solvers.L2NormSquared(data.space).translated(data) * forward_op
##odl.solvers.conjugate_gradient_nonlinear(func, reco, line_search=1e0, callback=callback,
##                                         nreset=50)
#
#
#lin_at_one = forward_op.derivative(forward_op.domain.one())
#backprop = lin_at_one.adjoint(data)
