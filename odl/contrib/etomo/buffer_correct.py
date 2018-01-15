import numpy as np

__all__ = ('buffer_correction',)


def buffer_correction(data, coords=[[0, 0.25], [0, 0.25]]):
    """Return data divided by the buffer contribution.

    Parameters
    ----------
    data : `DiscreteLpElement`
        The data to be corrected.
    coords : list of lists of floats, optional
       Specifies a region of the domain of 'data' that is considered to belong
       to the background. See the implementation for details.

    Returns
    ----------
    corrected_data : `DiscreteLpElement`
        The original data divided by the computed mean background, i.e.
        the buffer contribution.

    Notes
    ----------
    The forward model for TEM consist of operators that are either linear or
    multiplicative. It follows that the constant scattering potential of the
    buffer will contribute with a multiplicative constant to the data.
    """
    dim_t, dim_x, dim_y = data.shape
    data_asarray = data.asarray()

    # Pick out background according to coords
    bg = data_asarray[:, round(dim_x*coords[0][0]):round(dim_x*coords[0][1]),
                      round(dim_y*coords[1][0]):round(dim_y*coords[1][1])]

    bg_slice = bg[round(dim_t/2),:,:]
    import matplotlib.pyplot as plt
    plt.imshow(bg_slice)

    # Compute mean of background for all tomographic acquisition angles
    bg_mean = np.mean(bg, (1, 2))
    print(bg_mean.shape)

    # Return original data divided by the computed mean background
    return data.space.element(data_asarray * (1.0/bg_mean.reshape(dim_t,
                                                                  1, 1)))
