#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:23:11 2017

@author: zickert
"""

import numpy as np
import odl
from odl.contrib.electron_tomo.cast_operator import CastOperator
from random import shuffle, randint


def make_kaczmarz_plan(num_blocks, method='random',
                       num_blocks_per_superblock=1):

    indices = list(range(num_blocks))
    num_super_blocks = (num_blocks + (num_blocks_per_superblock-1)) // num_blocks_per_superblock
    if num_blocks % num_blocks_per_superblock != 0:
        indices.extend(indices[: num_blocks_per_superblock - (num_blocks % num_blocks_per_superblock)])

    block_indices = [indices[j*num_blocks_per_superblock: (j+1)*num_blocks_per_superblock]
                     for j in range(num_super_blocks)]

    if method == 'random':          # Randomized ordering
        shuffle(block_indices)
    elif method == 'mls':           # (M)ulti-(L)evel (S)cheme by Guan and Gordon (2005)
        block_indices = get_mls_order(block_indices)
    elif method == 'sequential':    # Retain sequential ordering
        pass

    return block_indices


# Sorts a list of tomographic angles (ideally between 0 and 180 degrees)
# such that the order represents the (M)ulti-(L)evel (S)cheme proposed by
#
# Guan and Gordon. Physics in medicine and biology 39.11 (1994): 2005.
#
# as a processing order for tomographic projections in Kaczmarz/ART-
# iterations.
def get_mls_order(tomo_angle_list, start_idx = None):

    length = len(tomo_angle_list)

    # Assign random start-index if not given
    if start_idx is None:
        start_idx = randint(0, length-1)

    # Extend list to have a length of 2^p for some integer p
    pow2_length = next_greater_power_of_2(length)
    pow2 = pow2_length.bit_length()-1
    #tomo_angle_list.extend(tomo_angle_list[:pow2_length-length])

    # Compute MLS-order for list of size 2^p
    order = np.arange(pow2_length)
    for jj in range(pow2):
        # divide-and-conquer-shuffling 
        order = np.reshape(order, (2**jj, pow2_length/2**(jj+1), 2))
        order = np.transpose(order, (0,2,1))
        order = np.reshape(order, (pow2_length,) )
    
    # Smartly truncate to get quasi MLS-order for length != 2^p
    order = np.round((length/(1.*pow2_length)) * order).astype(np.int32)    # Map indices to given length
    order = (order + start_idx) % length                                    # Set start index

    # Ensure that every index occurs exactly once (annoying two-liner
    # since numpy.unique likes to sort the array, which is not desired here)
    _, unique_idx = np.unique(order, return_index=True) 
    order = np.delete(order, np.setdiff1d(np.arange(pow2_length), unique_idx))
    
    return np.array(tomo_angle_list)[order].tolist()


def make_Op_blocks(block_indices, Block_Op, Op_pre=None, Op_post=None):

    if Op_pre is not None:
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


def make_data_blocks(data, block_indices, block_axis=0):

    def get_data_block(idx):
        return data.asarray()[block_indices[idx]]
    
    return get_data_block


# For a given integer n, computes the smallest m such that m>=n
# m = 2^p for some integer p
def next_greater_power_of_2(x):  
        return 2**(x-1).bit_length()



if __name__ == '__main__':
    print(get_mls_order(list(range(180))))
    print(get_mls_order([[2*jj, 2*jj+1] for jj in range(27)]))

