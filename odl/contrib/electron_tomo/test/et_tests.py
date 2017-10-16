import pytest


from odl.util.testutils import simple_fixture
from odl.contrib.electron_tomo.constant_phase_abs_ratio import ConstantPhaseAbsRatio


reco_space = simple_fixture('reco_space', [odl.uniform_discr(0, 1, 10)])


def test_constant_phase_abs_ratio(reco_space):
    ratio_op = ConstantPhaseAbsRatio(reco_space)

    x_test = reco_space.one()
    assert (1+1j)*ratio_op.range.element(x_test) == ratio_op(x_test)

    y_test = (2+1j)*ratio_op.range.one()
    assert 3*ratio_op.domain.one() == ratio_op.adjoint(y_test)

    x_adj = odl.phantom.white_noise(reco_space)
    y_adj = odl.phantom.white_noise(ratio_op.range)

    Ax_adj = ratio_op(x_adj)
    ATy_adj = ratio_op.adjoint(y_adj)

    ip1 = x_adj.inner(ATy_adj)
    ip2 = Ax_adj.inner(y_adj)

    assert pytest.approx(ip1.real) == ip2.real

def test_adjoint_relation_forward_op(reco_space):
    


if __name__ == '__main__':
    odl.util.test_file(__file__)
