'''
Evaluate trained PredNet on t_5 mode
Calculates mean-squared error and plots predictions
'''
import importlib
import numpy as np
# import tensorflow as tf
import random as rn
import os
import hickle as hkl
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from six.moves import cPickle
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from data_processing.data_utils import SequenceGenerator
from evaluation import evaluate_metrics_util
import torch
import pdb
# np.random.seed(123)
# rn.seed(123)
# tf.set_random_seed(123)

def separate_input_masks(tensors):
    # Check input shape but should be (?, nt, 128, 128, 3)
    dynamic_mask = tensors[:,:,:,:,-1].unsqueeze(-1)
    out_dynamic = torch.multiply(tensors[:,:,:,:,0:2], dynamic_mask)
    out_static = torch.multiply(tensors[:,:,:,:,0:2], 1-dynamic_mask)
    return [out_static, out_dynamic]


def full_grid_update(tensors):
    m_static = tensors[0]  # should be (None, 20, 128, 128, 2)
    m_dynamic = tensors[1]  # should be (None, 20, 128, 128, 2)
    print("m_static.shape", m_static.shape)
    print("m_dynamic.shape", m_dynamic.shape)
    # shpae of m_static_u should be (None, 20, 128, 128)
    m_static_u = 1 - m_static[:, :, :, :, 0] - m_static[:, :, :, :, 1]
    m_dynamic_u = 1 - m_dynamic[:, :, :, :, 0] - m_dynamic[:, :, :, :, 1]
    # new
    K = torch.multiply(m_static[:,:,:,:,0], m_dynamic[:,:,:,:,1]) - torch.multiply(m_static[:,:,:,:,1], m_dynamic[:,:,:,:,0])
    K = torch.minimum(K, torch.tensor([0.99]))
    denominator = 1 - K
    m_o = torch.div(torch.multiply(m_static[:,:,:,:,0], m_dynamic[:,:,:,:,0]) + torch.multiply(m_static_u,m_dynamic[:,:,:,:,0]) + torch.multiply(m_static[:, :, :, :, 0], m_dynamic_u), denominator)
    m_f = torch.div(torch.multiply(m_static[:,:,:,:,1], m_dynamic[:,:,:,:,1]) + torch.multiply(m_static_u,m_dynamic[:,:,:,:,1]) + torch.multiply(m_static[:, :, :, :, 1], m_dynamic_u), denominator)
    m_o = m_o.unsqueeze(-1)
    m_f = m_f.unsqueeze(-1)
    m_full = torch.cat([m_o, m_f], dim=-1)  # shape should be (None, 20, 128, 128, 2)
    print("shape m_full", m_full.shape)
    return m_full

def eval(X_test, X_hat):

    X_test = X_test.cpu().numpy().astype(np.float32)
    X_hat = X_hat.cpu().numpy().astype(np.float32)
    # ProbOccupancy for full grids

    X_test_full_probO = np.expand_dims(0.5*(X_test[:,:,:,:,0]) + 0.5*(1.-X_test[:,:,:,:,1]), axis=-1)
    X_hat_full_probO = np.expand_dims(0.5*X_hat[:,:,:,:,0] + 0.5*(1-X_hat[:,:,:,:,1]), axis=-1) # (:,20,128,128,1)

    # ProbOccupancy for dynamic objects
    X_test_masked_probO = np.multiply(X_test_full_probO, np.expand_dims(X_test[...,-1], axis=-1))
    X_hat_masked_probO = np.multiply(X_hat_full_probO, np.expand_dims(X_test[...,-1], axis=-1))

    # Image similarity metric
    avg_score, ms_score = evaluate_metrics_util.ImageSimilarityMetric(X_test_full_probO[:,:,:,:,0], X_hat_full_probO[:,:,:,:,0], start_time = 0)

    # MSE metric
    mse_score, mse_per_frame = evaluate_metrics_util.MSE(X_test_full_probO[:,:,:,:,0], X_hat_full_probO[:,:,:,:,0], start_time = 0)

    # MSE dynamic
    msed_score, mse_per_frame = evaluate_metrics_util.MSE(X_test_masked_probO[:, :, :, :, 0], X_hat_masked_probO[:, :, :, :, 0], start_time=0)

    return avg_score, mse_score, msed_score
