"""Phase contrast TEM reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

import os
import odl
from intensity_op import IntensityOperator
from odl.contrib.mrc import (
    FileReaderMRC, FileWriterMRC, mrc_header_from_params)


def pupil_function(x, **kwargs):
    """Indicator function for the disc-shaped aperture, a.k.a. pupil function.


    Notes
    -----
    The pupil fuction is defined as

    .. math::

        A_\\Sigma(\\xi) = \\begin{cases}1,& \\Big\\|\\frac{f}{k}\\xi\\Big\\|
        \\leq r\\\\ 0,& \\Big\\|\\frac{f}{k}\\xi\\Big\\| > r \\end{cases}

    where :math:`\kappa` is the wave number of the incoming electron wave,
    :math:`r` is the radius of the aperture and :math:`f` is the focal length
    of the principal lens.
    """
    aper_rad = kwargs.pop('aper_rad')
    focal_length = kwargs.pop('focal_length')
    wave_number = kwargs.pop('wave_number')
    scaled_rad = aper_rad * wave_number / focal_length

    norm_sq = np.sum(xi ** 2 for xi in x[1:])

    return norm_sq <= scaled_rad ** 2


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
    result = - (1 / (4 * wave_number)) * norm_sq * (norm_sq * spherical_abe /
                                                    wave_number ** 2 - 2 *
                                                    defocus)
    result = np.exp(1j * result)

    return result


def modulation_transfer_function(x, **kwargs):
    """Function that characterizes the detector response.


    Notes
    -----
    The modulation transfer function is defined as

    .. math::

        MTF(\\xi) = \\frac{a}{1 + \\alpha\\|\\xi\\|^2} +
        \\frac{b}{1+\\beta\\|\\xi\\|^2}+c

    where :math:`a,b,c,\\alpha` and :math:`\\beta` are real parameters."""
    a = kwargs.pop('mtf_a')
    b = kwargs.pop('mtf_b')
    c = kwargs.pop('mtf_c')
    alpha = kwargs.pop('mtf_alpha')
    beta = kwargs.pop('mtf_beta')

    norm_sq = np.sum(xi ** 2 for xi in x[1:]) / M**2

    result = a / (1 + alpha * norm_sq) + b / (1 + beta * norm_sq) + c

    return result


# %%
dir_path = os.path.abspath('/home/zickert/TEM_reco_project/One_particle_new_simulation')
file_path_phantom = os.path.join(dir_path, 'rna_phantom.mrc')
file_path_phantom_abs = os.path.join(dir_path, 'rna_phantom_abs.mrc')
file_path_tiltseries = os.path.join(dir_path, 'tiltseries.mrc')
file_path_tiltseries_nonoise = os.path.join(dir_path, 'tiltseries_nonoise.mrc')

with FileReaderMRC(file_path_phantom) as phantom_reader:
    phantom_header, phantom_asarray = phantom_reader.read()
with FileReaderMRC(file_path_phantom_abs) as phantom_abs_reader:
    phantom_abs_header, phantom_abs_asarray = phantom_abs_reader.read()
with FileReaderMRC(file_path_tiltseries) as tiltseries_reader:
    tiltseries_header, data_asarry = tiltseries_reader.read()
with FileReaderMRC(file_path_tiltseries_nonoise) as tiltseries_nonoise_reader:
    tiltseries_nonoise_header, data_nonoise_asarray = tiltseries_nonoise_reader.read()


#  Define some physical constants
e_mass = 9.11e-31  # kg
e_charge = 1.602e-19  # C
planck_bar = 1.059571e-34  # Js/rad

wave_length = 0.0025e-9  # m
wave_number = 2 * np.pi / wave_length

sigma = e_mass * e_charge / (wave_number * planck_bar ** 2)

total_dose = 5000 * 1e18  # total electron dose per m^2
dose_per_img = total_dose / 61
gain = 80  # average nr of digital counts per incident electron

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

# Set size of detector pixels (before rescaling to account for magnification)
det_size = 16e-6  # m
det_area = det_size ** 2  # m^2

reco_space = odl.uniform_discr(min_pt=[-95e-9/2, -100e-9/2,
                                       -80e-9/2],
                               max_pt=[95e-9/2, 100e-9/2,
                                       80e-9/2],
                               shape=[95, 100, 80], dtype='complex128')

angle_partition = odl.uniform_partition(-np.pi/3, np.pi/3, 61)
detector_partition = odl.uniform_partition([-det_size/M * 200/2] * 2,
                                           [det_size/M * 200/2] * 2, [200] * 2)

# The x-axis is the tilt-axis.
# Check that the geometry matches the one from TEM-simulator!
# In particular, check that det_pos_init and det_axes_init are correct.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=(1, 0, 0),
                                           det_pos_init=(0, 0, -1),
                                           det_axes_init=((1, 0, 0), (0, 1, 0)))
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
scattering_op = ray_trafo.range.one() + 1j*sigma * ray_trafo

ft_ctf = odl.trafos.FourierTransform(scattering_op.range, axes=[1, 2])
pupil = ft_ctf.range.element(pupil_function, aper_rad=aper_rad,
                             wave_number=wave_number,
                             focal_length=focal_length)

optics_imperf = ft_ctf.range.element(optics_imperfections,
                                     wave_number=wave_number,
                                     spherical_abe=spherical_abe,
                                     defocus=defocus)
ctf = 2 * np.pi * pupil * optics_imperf
optics_op_cst = 1/(M*(2*np.pi)**2)
optics_op = optics_op_cst * ft_ctf.inverse * ctf * ft_ctf

intens_op = IntensityOperator(optics_op.range)

# Check behaviour of the MTF
ft_det = odl.trafos.FourierTransform(intens_op.range, axes=[1, 2])
mtf = ft_det.range.element(modulation_transfer_function, mtf_a=mtf_a,
                           mtf_b=mtf_b, mtf_c=mtf_c, mtf_alpha=mtf_alpha,
                           mtf_beta=mtf_beta)
det_op = det_area * dose_per_img * gain * ft_det.inverse * mtf * ft_det

forward_op = det_op * intens_op * optics_op * scattering_op


phantom = reco_space.element(phantom_asarray + 1j * phantom_abs_asarray)
data = forward_op(phantom)

reco = ray_trafo.domain.zero()

lin_at_one = forward_op.derivative(forward_op.domain.one())
backprop = lin_at_one.adjoint(data)
