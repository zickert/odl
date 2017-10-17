#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:37:29 2017

@author: zickert
"""

import numpy as np
import odl
from odl.contrib.electron_tomo.intensity_op import IntensityOperator
from odl.contrib.electron_tomo.exp_operator import ExpOperator

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
    det_size = kwargs.pop('det_size')
    magnification = kwargs.pop('magnification')

    norm_sq = np.sum(xi ** 2 for xi in x[1:])
    # Rescale the length of the vector to account for larger detector in this
    # 2D toy example
    norm_sq *= (30 / (det_size / magnification * 100)) ** 2
    result = - (1 / (4 * wave_number)) * norm_sq * (norm_sq * spherical_abe /
                                                    wave_number ** 2 - 2 *
                                                    defocus)
    result = np.exp(1j * result)

    return result

def make_imageFormationOp(domain, wave_number, spherical_abe, defocus, det_size,
                          magnification, obj_magnitude=1.0, **kwargs):
    
    exp_op = ExpOperator(domain)
    
    ft_ctf = odl.trafos.FourierTransform(exp_op.range, axes=list(range(1,domain.ndim)))
    
    optics_imperf = ft_ctf.range.element(optics_imperfections,
                                         wave_number=wave_number,
                                         spherical_abe=spherical_abe,
                                         defocus=defocus,
                                         det_size=det_size,
                                         magnification=magnification)
    
    # Leave out pupil-function since it has no effect
    ctf = optics_imperf
    optics_op = ft_ctf.inverse * ctf * ft_ctf
    intens_op = IntensityOperator(optics_op.range)

    return intens_op * optics_op * exp_op * ((1j*obj_magnitude) * odl.IdentityOperator(exp_op.domain))
    