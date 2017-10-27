"""Phase contrast TEM reconstruction example."""


import numpy as np
import odl
from timeit import timeit
from odl.contrib import etomo


def circular_mask(x, **kwargs):
    radius = kwargs.pop('radius')
    norm_sq = np.sum(xi ** 2 for xi in x[:])

    return norm_sq <= radius ** 2


obj_magnitude = 1e-2
noise_lvl = 1e-1
num_angles = 120
regpar_TV = 1e-12
regpar_L2 = 1e-12
grad_factor = 1e-4

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
ray_trafo = etomo.BlockRayTransform(reco_space, geometry)

imageFormation_op = etomo.make_imageFormationOp(ray_trafo.range, 
                                                wave_number, spherical_abe,
                                                defocus, det_size, M,
                                                obj_magnitude=obj_magnitude)

mask = reco_space.element(etomo.spherical_mask, radius=19)

forward_op = imageFormation_op * ray_trafo * mask

f_op_lin = forward_op.derivative(reco_space.zero())

phantom = odl.phantom.shepp_logan(reco_space, modified=True)


data = f_op_lin(phantom)
noise = odl.phantom.white_noise(data.space)
data += (noise_lvl * data.norm() / noise.norm()) * noise

data.show()


# Initialize gradient operator
gradient = grad_factor * odl.Gradient(reco_space)

# Column vector of two operators
op = odl.BroadcastOperator(f_op_lin, gradient)

# Do not use the g functional, set it to zero.
g = regpar_L2 * odl.solvers.L2NormSquared(f_op_lin.domain) #odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(f_op_lin.range).translated(data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = (regpar_TV/grad_factor) * odl.solvers.L1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
f = odl.solvers.SeparableSum(l2_norm, l1_norm)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 28.4  # 1.1 * odl.power_method_opnorm(op)

niter = 200  # Number of iterations
sigma = 1 / op_norm  # Step size for the dual variable
tau = 1 / sigma / op_norm **2  # Step size for the primal variable
#tau = reco_space.one().norm() / (2.0*regpar_L2)
#sigma = (1.0 / tau) / op_norm **2

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Choose a starting point
x = 0.01*phantom

# Run the algorithm
odl.solvers.pdhg(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
                 callback=callback) #, gamma_primal = 2*regpar_L2
