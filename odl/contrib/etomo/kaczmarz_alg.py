import odl
import numpy as np
from odl.contrib.etomo.cast_operator import CastOperator
from odl.contrib.etomo.block_ray_trafo import BlockRayTransform
from odl.contrib.etomo.kaczmarz_util import make_kaczmarz_plan
from odl.contrib.etomo.kaczmarz_util import make_data_blocks
from odl.contrib.etomo.kaczmarz_util import make_Op_blocks


__all__ = ('kaczmarz_reco_method', 'kaczmarz_SART_method')


def kaczmarz_reco_method(get_Op, reco, get_data, num_iterates_per_cycle,
                         regpar, num_cycles=1, callback=None,
                         niter_CG=10, projection=None):

    for cycles in range(num_cycles):

        for it in range(num_iterates_per_cycle):

            op = get_Op(it)
            data = op.range.element(get_data(it))

            residual = data - op(reco)

#            if do_CG_on_data_space:
#
#                A = op.derivative(reco)
#                B = odl.IdentityOperator(A.range)
#                T = A * A.adjoint + regpar * B.adjoint * B
#
#                d_reco_T = residual.space.zero()
#                odl.solvers.conjugate_gradient(T, d_reco_T, residual,
#                                               niter=niter_CG)
#                d_reco = A.adjoint(d_reco_T)

            A = op.derivative(reco)
            B = odl.IdentityOperator(reco.space)
            T = A.adjoint * A + regpar * B.adjoint * B
            b = A.adjoint(residual)

            d_reco = reco.space.zero()
            odl.solvers.conjugate_gradient(T, d_reco, b, niter=niter_CG)

            reco += d_reco

            if projection is not None:
                projection(reco)

            if callback is not None:
                callback(reco)


def kaczmarz_SART_method(get_ProjOp, reco, get_data, num_iterates_per_cycle,
                         regpar, gamma_H1=0, imageFormationOp=None,
                         num_cycles=1, callback=None, niter_CG=10,
                         projection=None):

    for cycles in range(num_cycles):

        for it in range(num_iterates_per_cycle):

            ray_trafo = get_ProjOp(it)

            if imageFormationOp is None:
                imageFormOp = odl.IdentityOperator(ray_trafo.range)
            else:
                imageFormOp = imageFormationOp * CastOperator(ray_trafo.range,
                                                              imageFormationOp.domain)

            data = imageFormOp.range.element(get_data(it))

            # Forward project and compute residual
            p_reco = ray_trafo(reco)
            residual = data - imageFormOp(p_reco)

            # Compute unit-projection
            unit_proj = ray_trafo(ray_trafo.domain.one())
            unit_proj_sqrt = np.sqrt(unit_proj)

            if imageFormationOp is not None or gamma_H1 > 0:

                # Assemble operators for optimization in projection-space
                A = imageFormOp.derivative(p_reco)
                if gamma_H1 > 0:
                    # Regularize the unit-projection for numerical stability
                    unit_proj_np = unit_proj.asarray()
                    unit_proj_max = np.max(unit_proj_np)
                    unit_proj_supp = (unit_proj_np >= 1e-2*unit_proj_max)
                    unit_proj_np[unit_proj_np < 1e-6*unit_proj_max] = 1e-6*unit_proj_max
                    
                    # Define derivative such that it only acts within the support
                    unit_proj_supp_dd1 = p_reco.space.element(unit_proj_supp * np.roll(unit_proj_supp, -1, axis=1))
                    dd1 = unit_proj_supp_dd1 * odl.PartialDerivative(p_reco.space, 1)
                    
                    dd1_log_unit_proj = dd1(0.5 * np.log(unit_proj))
                    dd1_proj = dd1 - odl.MultiplyOperator(dd1_log_unit_proj, domain=dd1.domain, range=dd1.range)
                    if p_reco.space.ndim > 2:
                        unit_proj_supp_dd2 = p_reco.space.element(unit_proj_supp * np.roll(unit_proj_supp, -1, axis=2))
                        dd2 = unit_proj_supp_dd2 * odl.PartialDerivative(p_reco.space, 2)
                        dd2_log_unit_proj = dd2(0.5 * np.log(unit_proj))
                        dd2_proj = dd2 - odl.MultiplyOperator(dd2_log_unit_proj, domain=dd2.domain, range=dd2.range)
                        grad_proj = odl.BroadcastOperator(dd1_proj, dd2_proj)
                    else:
                        grad_proj = dd1_proj 
                    T = (unit_proj_sqrt * (A.adjoint * (A * unit_proj_sqrt))) + (regpar*(1.0-gamma_H1)) * odl.IdentityOperator(p_reco.space) + (regpar*gamma_H1) * (grad_proj.adjoint * grad_proj)
                else:
                    T = (unit_proj_sqrt * (A.adjoint * (A * unit_proj_sqrt))) + regpar * odl.IdentityOperator(p_reco.space)

                b = unit_proj_sqrt * A.adjoint(residual)

                # Solve for increment in projection
                d_p_tilde = T.domain.zero()
                odl.solvers.conjugate_gradient(T, d_p_tilde, b, niter=niter_CG)

                # Apply inverse preconditioner (regularized for numerical stability)
                unit_proj_sqrt_np = unit_proj_sqrt.asarray()
                unit_proj_sqrt_max = np.max(unit_proj_sqrt_np)
                unit_proj_sqrt_np[unit_proj_sqrt_np < 1e-2*unit_proj_sqrt_max] = 1e-2*unit_proj_sqrt_max
                d_p_tilde /= unit_proj_sqrt

            else:

                d_p_tilde = residual / (unit_proj + regpar)

            # Back-project and increment
            #backproj_correction_factor = ray_trafo.range.cell_volume/(ray_trafo.domain.cell_volume*ray_trafo.range.cell_sides[0])
            backproj_correction_factor = 1.0/ray_trafo.range.cell_sides[0]
#            unit_proj_2 = backproj_correction_factor * ray_trafo(ray_trafo.adjoint(ray_trafo.range.one()))
#            unit_proj_2.show()
#            unit_proj.show()

            reco += ray_trafo.adjoint(backproj_correction_factor * d_p_tilde)

            if projection is not None:
                projection(reco)

            if callback is not None:
                callback(reco)


if __name__ == '__main__':

    num_angles = 360
    num_angles_per_Kaczmarz_block = 1
    num_cycles = 3
    regpar = 1e-1

    reco_space = odl.uniform_discr(
        min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300])

    angle_partition = odl.uniform_partition(0, np.pi, num_angles)
    detector_partition = odl.uniform_partition(-30, 30, 512)

    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    block_ray_trafo = BlockRayTransform(reco_space, geometry)

    phantom = odl.phantom.shepp_logan(reco_space, modified=True)
    phantom.show()

    # Compute and show full data
    data = block_ray_trafo(phantom)
    data.show()

    callback = (odl.solvers.CallbackPrintIteration() &
                odl.solvers.CallbackShow())

    kaczmarz_plan = make_kaczmarz_plan(num_angles,
                                       num_blocks_per_superblock=num_angles_per_Kaczmarz_block,
                                       method='random')
    get_op = make_Op_blocks(kaczmarz_plan, block_ray_trafo)
    get_data = make_data_blocks(data, kaczmarz_plan)

    reco = reco_space.zero()
    #  kaczmarz_reco_method(get_op, reco, get_data, len(kaczmarz_plan), regpar, callback=callback, num_cycles=num_cycles)
    kaczmarz_SART_method(get_op, reco, get_data, len(kaczmarz_plan), regpar,
                         imageFormationOp=odl.IdentityOperator(get_op(0).range),
                         callback=callback, num_cycles=num_cycles)
