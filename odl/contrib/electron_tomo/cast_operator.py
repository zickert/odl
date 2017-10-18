import odl
from odl.operator.operator import Operator
from odl.discr.lp_discr import DiscreteLp
import numpy as np

__all__ = ('CastOperator',)


class CastOperator(Operator):

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

        super().__init__(domain, range, linear=True)

    def _call(self, x):
        return self.range.element(x.asarray())

    @property
    def adjoint(self):
         return CastOperator(domain = self.range, range = self.domain)

    @property
    def inverse(self):
         return CastOperator(domain = self.range, range = self.domain)
     
        
if __name__ == '__main__':
    
    cast_domain = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300])
    cast_range = cast_domain.complex_space
    
    cast = CastOperator(cast_domain, cast_range)
    
    one_cplx = cast(cast_domain.one())
