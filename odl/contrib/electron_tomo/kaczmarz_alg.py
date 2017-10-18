import odl
import numpy as np
from odl.contrib.electron_tomo.cast_operator import CastOperator


def kaczmarz_reco_method(get_Op, reco, get_data, num_iterates_per_cycle,
                         regpar, num_cycles = 1, callback = None,
                         niter_CG = 10, projection = None):
    
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
#                odl.solvers.conjugate_gradient(T, d_reco_T, residual, niter=niter_CG)
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
                         regpar, imageFormationOp = None, num_cycles = 1,
                         callback = None, niter_CG = 10, projection = None):
    
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
            
#            # Assemble operators for optimization in projection-space
#            A = imageFormOp.derivative(p_reco)
#            T = (unit_proj_sqrt * (A.adjoint * (A * unit_proj_sqrt))) + regpar * odl.IdentityOperator(p_reco.space)
#            b = unit_proj_sqrt * A.adjoint(residual)
#            
#            # Solve for increment in projection
#            d_p_tilde = T.domain.zero()
#            odl.solvers.conjugate_gradient(T, d_p_tilde, b, niter=niter_CG)
#            
#            # Apply inverse preconditioner
#            unit_proj_sqrt_np = unit_proj_sqrt.asarray()
#            unit_proj_sqrt_np[unit_proj_sqrt_np < 1e-5] = 1.0
#            #unit_proj_sqrt = unit_proj_sqrt.space.element(unit_proj_aa)
#            d_p_tilde /= unit_proj_sqrt
            
            d_p_tilde = residual / (unit_proj + regpar)
            
            # Back-project and increment
            reco += ray_trafo.adjoint((1.0 / ray_trafo.range.cell_sides[0]) * d_p_tilde)
            
            if projection is not None:
                projection(reco)
    
            if callback is not None:
                callback(reco)              
                
                
                
if __name__ == '__main__':
    import numpy as np
    from odl.contrib.electron_tomo.kaczmarz_util import *
    from odl.contrib.electron_tomo.block_ray_trafo import BlockRayTransform
    
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
    
    phantom = odl.phantom.shepp_logan(reco_space,modified=True)
    phantom.show() 
    
    # Compute and show full data
    data = block_ray_trafo(phantom)
    data.show()

    callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())
    
    kaczmarz_plan = make_kaczmarz_plan(num_angles, num_blocks_per_superblock = num_angles_per_Kaczmarz_block, method='random')
    get_op = make_Op_blocks(kaczmarz_plan, block_ray_trafo)
    get_data = make_data_blocks(data, kaczmarz_plan)
    
    reco = reco_space.zero()
    kaczmarz_reco_method(get_op, reco, get_data, len(kaczmarz_plan), regpar, callback=callback, num_cycles=num_cycles)
    #kaczmarz_SART_method(get_op, reco, get_data, len(kaczmarz_plan), regpar, callback=callback, num_cycles=num_cycles)