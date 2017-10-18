import odl


def kaczmarz_reco_method(get_Op, reco, get_data, num_iterates_per_cycle,
                         regpar, num_cycles = 1, callback = None,
                         niter_CG = 10, projection = None):
    
    for cycles in range(num_cycles):
        
        for it in range(num_iterates_per_cycle):
            
            op = get_Op(it)
            data = op.range.element(get_data(it))
            
            residual = data - op(reco)
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
                
                
if __name__ == '__main__':
    import numpy as np
    from odl.contrib.electron_tomo.kaczmarz_util import *
    from odl.contrib.electron_tomo.block_ray_trafo import BlockRayTransform
    
    num_angles = 360
    num_angles_per_Kaczmarz_block = 6
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
    
    kaczmarz_plan = make_kaczmarz_plan(num_angles, num_blocks_per_superblock = num_angles_per_Kaczmarz_block)
    get_op = make_Op_blocks(kaczmarz_plan, block_ray_trafo)
    get_data = make_data_blocks(data, kaczmarz_plan)
    
    reco = reco_space.zero()
    kaczmarz_reco_method(get_op, reco, get_data, len(kaczmarz_plan), regpar, callback=callback,num_cycles=1)