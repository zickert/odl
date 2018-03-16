import odl
from odl.tomo import RayTransform
import numpy as np

__all__ = ('BlockRayTransform',)


class BlockRayTransform(RayTransform):
    """Ray transform as a block operator."""

    def __init__(self, domain, geometry, **kwargs):
        super(BlockRayTransform, self).__init__(domain, geometry, **kwargs)

    def get_sub_operator(self, sub_op_idx):
        """Return a block of the ray transform."""
        
        if np.size(sub_op_idx) != 1:
        
            return odl.tomo.RayTransform(self.domain,
                                         self.geometry[sub_op_idx])
        else:
            # Hack for defining Ray-trafo with single-angle geometry
            angle = self.geometry.angles[sub_op_idx[0]]
            
            if sub_op_idx[0] != self.geometry.angles.shape[0] -1:
                min_pt = angle
                max_pt = self.geometry.angles[sub_op_idx[0]+1]
            else:
                min_pt = angle
                max_pt = angle + (angle - self.geometry.angles[sub_op_idx[0]-1])
#            angle_increment = self.geometry.angles[1] - self.geometry.angles[0]
            apart = odl.uniform_partition(min_pt, max_pt, 1,
                                          nodes_on_bdry=(True,False))
            dpart = self.geometry.det_partition
            geom = odl.tomo.Parallel3dAxisGeometry(apart,
                                                   dpart,
                                                   axis=self.geometry.axis,
                                                   det_pos_init=self.geometry._det_pos_init_arg,
                                                   det_axes_init=self.geometry._det_axes_init_arg,
                                                   translation=self.geometry.translation)
            return odl.tomo.RayTransform(self.domain, geom)

if __name__ == '__main__':
    import numpy as np

    reco_space = odl.uniform_discr(
        min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300])

    angle_partition = odl.uniform_partition(0, np.pi, 360)
    detector_partition = odl.uniform_partition(-30, 30, 512)

    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    block_ray_trafo = BlockRayTransform(reco_space, geometry)

    phantom = odl.phantom.shepp_logan(reco_space, modified=True)
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
                          omega=1.0/(40*360//subsample_rate),
                          callback=callback)
