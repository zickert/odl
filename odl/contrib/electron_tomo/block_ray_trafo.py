#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:46:09 2017

@author: zickert
"""
import odl
from odl.tomo import RayTransform

class BlockRayTransform(RayTransform):
    
    def __init__(self, domain, geometry, **kwargs):
        super(BlockRayTransform, self).__init__(domain, geometry, **kwargs)
        
    def get_sub_operator(self, sub_op_idx):
        return odl.tomo.RayTransform(self.domain, self.geometry[sub_op_idx])


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
    
    # Compute and show partial data
    subsample_rate = 10
    sub_idx = [subsample_rate*j for j in range(360//subsample_rate)]
    sub_op = block_ray_trafo.get_sub_operator(sub_idx)
    data_block = sub_op(phantom)
    data_block.show()
    
    callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())
    odl.solvers.landweber(sub_op, sub_op.domain.zero(), data_block, 1000, 
                          omega=1.0/(40*360//subsample_rate), callback=callback)