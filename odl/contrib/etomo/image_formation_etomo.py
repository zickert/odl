import numpy as np
import odl
from odl.contrib.etomo.exp_operator import ExpOperator
from odl.contrib.etomo.constant_phase_abs_ratio import ConstantPhaseAbsRatio
from odl.contrib.etomo.intensity_op import IntensityOperator

__all__ = ('pupil_function', 'modulation_transfer_function',
           'optics_imperfections', 'make_imageFormationOp')


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
    magnification = kwargs.pop('magnification')

    norm_sq = np.sum(xi ** 2 for xi in x[1:]) / magnification**2

    result = a / (1 + alpha * norm_sq) + b / (1 + beta * norm_sq) + c

    return result


def optics_imperfections(xi, **kwargs):
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
    axes = kwargs.pop('axes')
    rescale_factor = kwargs.pop('rescale_factor')

    norm_sq = np.sum(xi[dim] ** 2 for dim in axes)
    # Rescale to account for larger detector in toy examples
    # rescale_factor = (30 / (det_size / magnification * 100)) ** 2
    norm_sq *= rescale_factor ** 2
    result = - (1 / (4 * wave_number)) * norm_sq * (norm_sq * spherical_abe /
                                                    wave_number ** 2 - 2 *
                                                    defocus)
    result = np.exp(1j * result)

    return result


def make_imageFormationOp(domain, wave_number, spherical_abe, defocus,
                          abs_phase_ratio=1, obj_magnitude=1,
                          rescale_factor=1, **kwargs):

    ratio_op = ConstantPhaseAbsRatio(domain, abs_phase_ratio=abs_phase_ratio,
                                     magnitude_factor=obj_magnitude)

    # Create (pointwise) exponential operator.
    exp_op = ExpOperator(ratio_op.range)

    # Get axes on which to perform FT.
    ft_axes = list(range(1, domain.ndim))

    # Create (partial) FT
    ft_ctf = odl.trafos.FourierTransform(exp_op.range, axes=ft_axes)

    optics_imperf = ft_ctf.range.element(optics_imperfections,
                                         wave_number=wave_number,
                                         spherical_abe=spherical_abe,
                                         defocus=defocus, axes=ft_axes,
                                         rescale_factor=rescale_factor)

    # Leave out pupil-function in the ctf
    ctf = optics_imperf

    # The optics operator is a multiplication in frequency-space
    optics_op = ft_ctf.inverse * ctf * ft_ctf
    intens_op = IntensityOperator(optics_op.range)

    # optics_op_cst = 2*np.pi /(magnification*(2*np.pi)**2)
    # det_op_cst = det_area * dose_per_img * gain
    # total_cst = optics_op_cst ** 2 * det_op_cst

    # Check behaviour of the MTF
    # ft_det = odl.trafos.FourierTransform(intens_op.range, axes=[1, 2])
    # mtf = ft_det.range.element(modulation_transfer_function, mtf_a=mtf_a,
    #                           mtf_b=mtf_b, mtf_c=mtf_c, mtf_alpha=mtf_alpha,
    #                           mtf_beta=mtf_beta)
    # det_op_cst = det_area * dose_per_img * gain
    # det_op = det_op_cst * ft_det.inverse * mtf * ft_det

    return intens_op * optics_op * exp_op * ratio_op
