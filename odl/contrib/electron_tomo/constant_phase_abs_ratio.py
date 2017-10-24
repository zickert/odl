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

    def __init__(self, domain=None, range=None, abs_phase_ratio=1,
                 magnitude_factor=1):

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

        super(ConstantPhaseAbsRatio, self).__init__(domain, range, linear=True)

        self.abs_phase_ratio = abs_phase_ratio
        self.magnitude_factor = magnitude_factor
        self.embedding_factor = (1j - self.abs_phase_ratio)*magnitude_factor

    def _call(self, x):
        """Implement ``self(x, out)``."""
        return self.range.element(x)*self.embedding_factor

    @property
    def adjoint(self):

        class AbsPhaseAdj(Operator):
            def __init__(self, op):
                super(AbsPhaseAdj, self).__init__(op.range, op.domain, linear=True)
                self.embedding_factor_c = np.conj(op.embedding_factor)

            def _call(self, g):
                return self.range.element(np.real(self.embedding_factor_c*g))

        return AbsPhaseAdj(self)
