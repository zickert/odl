import odl
from odl.operator.operator import Operator
from odl.discr.lp_discr import DiscreteLp

__all__ = ('CastOperator',)


class CastOperator(Operator):

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

        super(CastOperator, self).__init__(domain, range, linear=True)

    def _call(self, x):
        return self.range.element(x.asarray())

    @property
    def adjoint(self):
        return CastOperator(domain=self.range, range=self.domain)

    @property
    def inverse(self):
        return CastOperator(domain=self.range, range=self.domain)
