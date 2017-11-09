from odl.operator.operator import Operator
from odl.discr.lp_discr import DiscreteLp
import numpy as np

__all__ = ('ExpOperator',)


class ExpOperator(Operator):

    """Operator mapping a function to its pointwise exponential."""

    def __init__(self, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`, optional
            The space of elements which the operator acts on. If
            ``range`` is given, ``domain`` must be a power space
            of ``range``.
        range : `DiscreteLp`, optional
            The space of elements to which the operator maps.
            This is required if ``domain`` is not given.

        Notes
        -----
        This operator maps a real-valued function to its pointwise exponential

        .. math::
            \\text{Exp}(f(\cdot)) = e^{f(\cdot)}.

        """
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
        """Implement ``self(x)``."""
        return np.exp(x)

    def derivative(self, f):
        """Return the derivative operator in ``f``.

        Parameters
        ----------
        f : domain element
            Point at which the derivative is taken

        Returns
        -------
        deriv : `Operator`
            Derivative operator at the specified point

        Notes
        -----
        The derivative of the exponential operator is given by

        .. math::
            \\left(\partial \\text{Exp}(f)\\right)(h) = \\text{Exp}(f)h

        Its adjoint is given by

        .. math::
             \\left[\partial \\text{Exp}(f)\\right]^*(g) =
             \\overline{\\text{Exp}(f)}g
        """
        exp_f = np.exp(f)
        op = self

        class ExpOpDeriv(Operator):
            def __init__(self):
                super(ExpOpDeriv, self).__init__(op.domain, op.range,
                                                 linear=True)

            def _call(self, h):
                return exp_f * h

            @property
            def adjoint(self):
                class ExpOpDerivAdj(Operator):
                    def __init__(self):
                        super(ExpOpDerivAdj, self).__init__(op.range,
                                                            op.domain,
                                                            linear=True)

                    def _call(self, g):
                        return exp_f.conj() * g

                return ExpOpDerivAdj()

        return ExpOpDeriv()
