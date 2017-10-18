# -*- coding: utf-8 -*-
import numpy as np


def spherical_mask(x, **kwargs):
    radius = kwargs.pop('radius')
    norm_sq = np.sum(xi ** 2 for xi in x[:])

    return norm_sq <= radius ** 2
