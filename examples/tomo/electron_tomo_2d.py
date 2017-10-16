"""Phase contrast TEM reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import odl
from intensity_op import IntensityOperator

def optics_imperfections(x, **kwargs):
    """Function encoding the phase shifts due to optics imperfections.


    Notes
    -----
    The optics-imperfections function is defined as

    .. math::

        O(\\xi) = e^{iW(||\\xi\\|^2)},

    where the function :math:`W` is defined as

    .. math::

        W(t) = -\\frac{1}{4k}t\\left(\\frac{C_s}{k^2}t-2\\Delta z\\right),

    and where :math:`\kappa` is the wave number of the incoming electron wave,
    :math:`C_s` is the third-order spherical abberation of the lens and
    :math:`\\Delta z` is the defocus."""
    wave_number = kwargs.pop('wave_number')
    spherical_abe = kwargs.pop('spherical_abe')
    defocus = kwargs.pop('defocus')

    norm_sq = np.sum(xi ** 2 for xi in x[1:])
    # Rescale the length of the vector to account for larger detector in this
    # 2D toy example
    norm_sq *= (30 / (det_size / M * 100)) ** 2
    result = - (1 / (4 * wave_number)) * norm_sq * (norm_sq * spherical_abe /
                                                    wave_number ** 2 - 2 *
                                                    defocus)
    result = np.exp(1j * result)

    return result


# %%
det_size = 16e-6  # m
wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length

# Define sample diameter and height (we take the height at the edge)
sample_diam = 1200e-9  # m
sample_height = 150e-9  # m

# Define properties of the optical system
# Set focal_length to be the focal_length of the principal (first) lens !
M = 25000.0
aper_rad = 0.5*40e-6  # m
focal_length = 2.7e-3  # m
spherical_abe = 2.1e-3  # m
defocus = 3e-6  # m

# Define constants defining the modulation transfer function
mtf_a = 0.7
mtf_b = 0.2
mtf_c = 0.1
mtf_alpha = 10
mtf_beta = 40

reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='complex128')

angle_partition = odl.uniform_partition(0, np.pi, 360)
detector_partition = odl.uniform_partition(-30, 30, 512)

geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Choose constant before ray_trafo so that the result is small enough for a
# linearisation of the exponential to make sense.
scattering_op = ray_trafo.range.one() + 0.01j * ray_trafo

ft_ctf = odl.trafos.FourierTransform(scattering_op.range, axes=1)

optics_imperf = ft_ctf.range.element(optics_imperfections,
                                     wave_number=wave_number,
                                     spherical_abe=spherical_abe,
                                     defocus=defocus)

# Leave out pupil-function since it has no effect
ctf = optics_imperf
optics_op = ft_ctf.inverse * ctf * ft_ctf

intens_op = IntensityOperator(optics_op.range)

# Leave out detector operator for simplicity
forward_op = intens_op * optics_op * scattering_op
forward_op = optics_op * scattering_op

phantom = (1+1j) * odl.phantom.shepp_logan(reco_space, modified=True)

data = forward_op(phantom)

reco = reco_space.zero()
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())
odl.solvers.conjugate_gradient_normal(forward_op, reco, data,
                                      niter=10, callback=callback)

# non-linear cg must be adapted to complex case
#func = odl.solvers.L2NormSquared(data.space).translated(data) * forward_op
#odl.solvers.conjugate_gradient_nonlinear(func, reco, line_search=1e0, callback=callback,
#                                         nreset=50)


lin_at_reco = forward_op.derivative(reco)
backprop = lin_at_reco.adjoint(data)

