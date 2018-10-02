import pytest
import odl
import numpy as np


from odl.util.testutils import simple_fixture
from odl.contrib.etomo.constant_phase_abs_ratio import ConstantPhaseAbsRatio
from odl.contrib.etomo.exp_operator import ExpOperator
from odl.contrib.etomo.cast_operator import CastOperator
from odl.util.testutils import all_almost_equal

reco_space = simple_fixture('reco_space', [odl.uniform_discr(0, 1, 10)])
reco_space_complex = simple_fixture('reco_space_complex',
                                    [odl.uniform_discr(0, 1, 10,
                                                       dtype='complex')])


def test_constant_phase_abs_ratio(reco_space):
    magnitude = np.random.rand()
    ratio = np.random.rand()
    ratio_op = ConstantPhaseAbsRatio(reco_space, abs_phase_ratio=ratio,
                                     magnitude_factor=magnitude)

    x_test = reco_space.one()
    expected_result = (1j-ratio)*magnitude*ratio_op.range.element(x_test)

    assert expected_result == ratio_op(x_test)

    y_test = (2+1j)*ratio_op.range.one()
    assert all_almost_equal((1-2*ratio)*magnitude*ratio_op.domain.one(),
                            ratio_op.adjoint(y_test))  # (-1j-ratio)*(2+1j))

    x_adj = odl.phantom.white_noise(reco_space)
    y_adj = odl.phantom.white_noise(ratio_op.range)

    Ax_adj = ratio_op(x_adj)
    ATy_adj = ratio_op.adjoint(y_adj)

    ip1 = x_adj.inner(ATy_adj)
    ip2 = Ax_adj.inner(y_adj)

    assert pytest.approx(ip1.real) == ip2.real


def test_exp_operator(reco_space_complex):
    x_test = (1+1j*np.pi/2)*reco_space_complex.one()
    exp_op = ExpOperator(reco_space_complex)
    assert all_almost_equal(exp_op(x_test),
                            reco_space_complex.one() * (1j*np.exp(1)))

    x_0 = odl.phantom.white_noise(reco_space_complex)

    exp_deriv = exp_op.derivative(x_0)

    x_adj = odl.phantom.white_noise(reco_space_complex)
    y_adj = odl.phantom.white_noise(exp_op.range)

    Ax_adj = exp_deriv(x_adj)
    ATy_adj = exp_deriv.adjoint(y_adj)

    ip1 = x_adj.inner(ATy_adj)
    ip2 = Ax_adj.inner(y_adj)

    assert pytest.approx(ip1.real) == ip2.real


def test_cast_operator(reco_space):

    cast_domain = reco_space
    cast_range = cast_domain.complex_space
    cast = CastOperator(cast_domain, cast_range)

    one_cplx = cast(cast_domain.one())

    assert all_almost_equal(one_cplx, cast_range.one())


if __name__ == '__main__':
    odl.util.test_file(__file__)

# Test for adjoint moved here from an example
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
