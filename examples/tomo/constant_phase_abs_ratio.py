#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:28:52 2017

@author: zickert
"""
import odl
from odl import Operator
from odl import DiscreteLp
import numpy as np


class ConstantPhaseAbsRatio(Operator):

    """Intensity mapping of a vectorial function."""

    def __init__(self, domain=None, range=None, abs_phase_ratio=1):
        """Initialize a new instance.

        Parameters
        ----------
        domain : power space of `DiscreteLp`, optional
            The space of elements which the operator acts on. If
            ``range`` is given, ``domain`` must be a power space
            of ``range``.
        range : `DiscreteLp`, optional
            The space of elements to which the operator maps.
            This is required if ``domain`` is not given.

        Notes
        -----
        This operator maps a complex vector field :math:`f = (f_1, \dots, f_d)`
        to its pointwise intensity

            :math:`\mathcal{I}(f) = \\lvert f\\rvert^2 :
            x \mapsto \sum_{j=1}^d \\lvert f_i(x)\\rvert ^2`.

        """
        if domain is None and range is None:
            raise ValueError('either domain or range must be specified.')

        if domain is None:
            if not isinstance(range, DiscreteLp):
                raise TypeError('range {!r} is not a DiscreteLp instance.'
                                ''.format(range))
            domain = range.real_space

        if range is None:
            if not isinstance(domain, DiscreteLp):
                raise TypeError('domain {!r} is not a `DiscreteLp` '
                                'instance.'.format(domain))
            range = domain.complex_space

        super().__init__(domain, range, linear=True)

        self.abs_phase_ratio = abs_phase_ratio

    def _call(self, x):
        """Implement ``self(x, out)``."""
        return self.range.element(x)*(1 + 1j*self.abs_phase_ratio)

    @property
    def adjoint(self):

        class AbsPhaseAdj(Operator):
            def __init__(self, op):
                super().__init__(op.range, op.domain, linear=True)
                self.abs_phase_ratio = op.abs_phase_ratio

            def _call(self, g):
                return self.range.element(np.real((1-1j*self.abs_phase_ratio)*g))

        return AbsPhaseAdj(self)


#  UNIT TESTS
if __name__ == '__main__':

    reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300])

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

    assert abs(ip1.real-ip2.real) < 1e-12*abs(ip1.real)
