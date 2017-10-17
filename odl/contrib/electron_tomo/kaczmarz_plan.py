#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:23:11 2017

@author: zickert
"""

import odl
from odl.contrib.electron_tomo.cast_operator import CastOperator
from random import shuffle


def make_kaczmarz_plan(num_blocks, method='sequential',
                       num_blocks_per_superblock=1):

    indices = list(range(num_blocks))
    num_super_blocks = (num_blocks + (num_blocks_per_superblock-1)) // num_blocks_per_superblock
    if num_blocks % num_blocks_per_superblock != 0:
        indices.extend(indices[ : num_blocks_per_superblock - (num_blocks % num_blocks_per_superblock)])

    block_indices = [indices[j*num_blocks_per_superblock: (j+1)*num_blocks_per_superblock]
                     for j in range(num_super_blocks)]

    if method == 'random':
        # TODO: shuffle indices
        shuffle(block_indices)
    elif method == 'sequential':
        pass

    return block_indices


def make_Op_blocks(block_indices, Block_Op, Op_pre = None, Op_post = None):

    if Op_post is not None:
        if Op_post is not None:
            def get_Op(idx):
                sub_op = Block_Op.get_sub_operator(block_indices[idx])
                return Op_post * CastOperator(sub_op.range, Op_post.domain) * sub_op * Op_pre
        else:
            def get_Op(idx):
                sub_op = Block_Op.get_sub_operator(block_indices[idx])
                return sub_op * Op_pre
    else:
        if Op_post is not None:
            def get_Op(idx):
                sub_op = Block_Op.get_sub_operator(block_indices[idx])
                return Op_post * CastOperator(sub_op.range, Op_post.domain) * sub_op
        else:
            def get_Op(idx):
                sub_op = Block_Op.get_sub_operator(block_indices[idx])
                return sub_op
                
    return get_Op

    
def make_data_blocks(data, block_indices, block_axis = 0):
    
    def get_data_block(idx):
        return data.asarray()[block_indices[idx]]
    
    return get_data_block
    

def kaczmarz_reco_method(get_Op, reco, get_data, num_iterates_per_cycle, regpar, 
                         num_cycles = 1, callback = None, niter_CG = 10, projection = None):
    
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

    reco_space = odl.uniform_discr(
        min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300])
    
    angle_partition = odl.uniform_partition(0, np.pi, 360)
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
    
    kaczmarz_plan = make_kaczmarz_plan(360, num_blocks_per_superblock = 6)
    get_op = make_Op_blocks(kaczmarz_plan, block_ray_trafo)
    get_data = make_data_blocks(data, kaczmarz_plan)
    
    reco = reco_space.zero()
    kaczmarz_reco_method(get_op, reco, get_data, len(kaczmarz_plan), 1.0, callback=callback,num_cycles=1)
