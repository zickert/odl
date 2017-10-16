#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:28:52 2017

@author: zickert
"""
from odl import Operator
from odl import DiscreteLp
import numpy as np

__all__ = ('ConstantPhaseAbsRatio',)


class ConstantPhaseAbsRatio(Operator):

    def __init__(self, domain=None, range=None, abs_phase_ratio=1):

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
