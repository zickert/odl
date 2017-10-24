from odl.operator.operator import Operator
from odl.discr.lp_discr import DiscreteLp
import numpy as np

__all__ = ('ExpOperator',)


class ExpOperator(Operator):

    """Intensity mapping of a vectorial function."""

    def __init__(self, domain=None, range=None):
        if domain is None and range is None:
            raise ValueError('either domain or range must be specified.')

        if domain is None:
            if not isinstance(range, DiscreteLp):
                raise TypeError('range {!r} is not a DiscreteLp instance.'
                                ''.format(range))
            domain = range

        if range is None:
            if not isinstance(domain, DiscreteLp):
                raise TypeError('domain {!r} is not a `DiscreteLp` '
                                'instance.'.format(domain))
            range = domain

        super(ExpOperator, self).__init__(domain, range, linear=False)

    def _call(self, x):
        """Implement ``self(x, out)``."""
        return np.exp(x)

    def derivative(self, f):
        
        exp_f = np.exp(f)
        op = self

        class ExpOpDeriv(Operator):
            def __init__(self):
                super(ExpOpDeriv, self).__init__(op.domain, op.range, linear=True)

            def _call(self, h):
                return exp_f * h

            @property
            def adjoint(self):
                class ExpOpDerivAdj(Operator):
                    def __init__(self):
                        super(ExpOpDerivAdj, self).__init__(op.range, op.domain, linear=True)

                    def _call(self, g):
                        return exp_f.conj() * g

                return ExpOpDerivAdj()

        return ExpOpDeriv()
