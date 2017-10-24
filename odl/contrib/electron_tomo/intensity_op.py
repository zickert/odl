from odl.operator.operator import Operator
from odl.discr.lp_discr import DiscreteLp

__all__ = ('IntensityOperator',)


class IntensityOperator(Operator):

    """Intensity mapping of a vectorial function."""

    def __init__(self, domain=None, range=None):
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
            domain = range.complex_space

        if range is None:
            if not isinstance(domain, DiscreteLp):
                raise TypeError('domain {!r} is not a `DiscreteLp` '
                                'instance.'.format(domain))
            range = domain.real_space

        super(IntensityOperator, self).__init__(domain, range, linear=False)

    def _call(self, x, out):
        """Implement ``self(x, out)``."""
        out[:] = x.real
        out *= out
        out += x.imag * x.imag

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
        The derivative of the intensity operator is given by

            :math:`\partial \mathcal{I}(f_1, f_2)(h_1, h_2) =
            2 (f_1 h_1 + f_2 h_2)`.

        Its adjoint maps a function :math:`g` to the product space
        element

            :math:`\\left[\partial\mathcal{I}(f_1, f_2)\\right]^*(g) =
            2 (f_1 g, f_2 g)`.
        """
        op = self

        class IntensOpDeriv(Operator):
            def __init__(self):
                super(IntensOpDeriv, self).__init__(op.domain, op.range, linear=True)

            def _call(self, h):
                return 2 * (f.real * h.real + f.imag * h.imag)

            @property
            def adjoint(self):
                class IntensOpDerivAdj(Operator):
                    def __init__(self):
                        super(IntensOpDerivAdj, self).__init__(op.range, op.domain, linear=True)

                    def _call(self, g, out):
                        out.real[:] = 2 * g * f.real
                        out.imag[:] = 2 * g * f.imag

                return IntensOpDerivAdj()

        return IntensOpDeriv()
