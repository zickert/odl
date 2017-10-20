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
        indices.extend(indices[: num_blocks_per_superblock - (num_blocks % num_blocks_per_superblock)])

    block_indices = [indices[j*num_blocks_per_superblock: (j+1)*num_blocks_per_superblock]
                     for j in range(num_super_blocks)]

    if method == 'random':
        # TODO: shuffle indices
        shuffle(block_indices)
    elif method == 'sequential':
        pass

    return block_indices


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
