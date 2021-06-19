# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:31:32 2020

@author: Rob
"""

import os
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count=48"
import numpy as np
import jax.numpy as jnp
import jax.ops
import jax.lax
import jax.random as random
from jax import jit
from jax import vmap
from jax import pmap
from jax.config import config
config.update("jax_enable_x64", True)

#%%

# ==========
# Parameters
# ==========

# system
d = 20 # dimensionality of spin chain
h = 4.0 # strength of z interactions
alpha = 5 # number of RBM features
iterations = 101 # number of parameter updates
parallel = 50 # number of parallel MC chains
T = 20 # number of steps per MC chain

# rgn
rgn = True
reg_max = .1
# sr
sr = True
# gd
gd = False

# random seeds
cores = jax.local_device_count()
print('Cores:', cores)
key = random.PRNGKey(123)
key_save = random.split(key, cores)

#%%

# ======
# Ansatz
# ======

@jit
def ansatz(state, features2, bias):
    state2 = jnp.fft.fft2(state)
    angles = jnp.fft.ifft2(features2*jnp.conj(state2))+ bias
    log_wave = jnp.sum(jnp.log(jnp.cosh(angles)))
    return(log_wave)
ansatz1 = vmap(ansatz, (0, None, None), 0)
jansatz = jit(ansatz1)

# ==============
# Local energies
# ==============

@jit
def e_locs(states, features2, bias):
    # spin-z
    diffs = states ^ states[..., (jnp.arange(d) + 1) % d, :]
    energies = 2*jnp.sum(diffs, axis = (-2, -1)) - d**2
    diffs = states ^ states[..., (jnp.arange(d) + 1) % d]
    energies += 2*jnp.sum(diffs, axis = (-2, -1)) - d**2
    # neighbor states
    states2 = jnp.expand_dims(states, axis = -3)
    states2 = jnp.repeat(states2, d**2 + 1, axis = -3)
    i0 = jnp.arange(d**2)
    i1 = jnp.repeat(jnp.arange(d), d)
    i2 = jnp.tile(jnp.arange(d), d)
    states2 = jax.ops.index_update(states2, jax.ops.index[..., i0, i1, i2], \
                                    ~states2[..., i0, i1, i2])
    # flip neighbors
    flips = (2*jnp.sum(states2, axis = (-2, -1)) + states2[..., 0, 0] > d**2)
    states2 = states2 ^ jnp.expand_dims(flips, (-2, -1))
    # fast evaluation
    locs = jloc(states2, energies, features2, bias)
    return(locs)

@jit
def loc(states, energy, features2, bias):
    log_waves = jansatz(states, features2, bias)
    wave_ratios = jnp.exp(log_waves[:-1] - log_waves[-1])
    local = energy - h*jnp.sum(wave_ratios)
    return(local)
loc1 = vmap(loc, (0, 0, None, None), 0)
jloc = jit(loc1)

# ========
# Sampling
# ========

@jit
def get_data(states, key, weights):
    # initialize
    bias = jnp.reshape(weights[-alpha:], (alpha, 1, 1))
    features = jnp.reshape(weights[:-alpha], (alpha, d, d))
    features2 = jnp.fft.fft2(features)
    currents = jansatz(states, features2, bias)
    # generate data
    (states, currents, key, _, _,), (store_energy) = \
        jax.lax.scan(sample_less, (states, currents, key, features2, bias), None, T)
    return(states, key, store_energy)
parallel_data = pmap(get_data, in_axes = (0, 0, None), 
                     out_axes = (0, 0, 0))

@jit
def sample_less(inputs, i):
    (states, currents, key, features2, bias), _ = \
        jax.lax.scan(update, inputs, None, d)
    locs = e_locs(states, features2, bias)
    return (states, currents, key, features2, bias), (locs)

@jit
def update(inputs, i):
    (states, currents, key, features2, bias) = inputs
    # randomly perturb
    key, key1, key2, key3, key4, key5 = random.split(key, num = 6)
    i0 = jnp.arange(parallel)
    i1 = random.choice(key1, d, shape = (parallel,))
    i2 = random.choice(key2, d, shape = (parallel,))
    perturbs = jax.ops.index_update(states, jax.ops.index[i0, i1, i2],
                                    ~states[i0, i1, i2])    
    # flip spins
    flips = (2*jnp.sum(perturbs, axis = (-2, -1)) + perturbs[..., 0, 0] > d**2)
    perturbs = perturbs ^ jnp.expand_dims(flips, (-2, -1))
    # accept or reject moves
    futures = jnp.real(jansatz(perturbs, features2, bias))
    accept = random.exponential(key3, shape = (parallel,))
    accept = (futures - currents > -.5*accept)
    # update information
    currents = currents * ~accept + futures * accept
    accept = accept[:, jnp.newaxis, jnp.newaxis]
    states = (states & ~accept) | (perturbs & accept)
    return (states, currents, key, features2, bias), None

#%%

# =====================
# MCMC testing -- RGN
# =====================

if rgn:

    # starting condition
    key = jnp.array(key_save)
    weights = jnp.load('nrgn_weights.npy')[-1, :]
    states = jnp.load('nrgn_save_1000.npy')
    rgn_est = np.zeros(iterations) + 0j

    # test the energy
    for iteration in range(iterations):
        print(iteration)
        (states, key, store_energy) = parallel_data(states, key, weights)
        rgn_est[iteration] = jnp.mean(store_energy)
        print('Estimated energy: ', rgn_est[iteration]/d**2)
        if iteration % 100 == 0:
            if iteration > 0:
                np.save('nrgn_test.npy', rgn_est)
                    
#%%

#===================
# MCMC training - SR
# ==================

if sr:

    # starting condition
    key = jnp.array(key_save)
    weights = jnp.load('sr_weights.npy')[-1, :]
    states = jnp.load('sr_save_1000.npy')
    sr_est = np.zeros(iterations) + 0j

    # test the energy
    for iteration in range(iterations):
        print(iteration)
        (states, key, store_energy) = parallel_data(states, key, weights)
        sr_est[iteration] = jnp.mean(store_energy)
        print('Estimated energy: ', sr_est[iteration]/d**2)
        if iteration % 100 == 0:
            if iteration > 0:
                np.save('sr_test.npy', sr_est)
                    
#%%

#===================
# MCMC training - GD
# ==================

if gd:

    # starting condition
    key = jnp.array(key_save)
    weights = jnp.load('gd_weights.npy')[-1, :]
    states = jnp.load('gd_save_1000.npy')
    gd_est = np.zeros(iterations) + 0j

    # test the energy
    for iteration in range(iterations):
        print(iteration)
        (states, key, store_energy) = parallel_data(states, key, weights)
        gd_est[iteration] = jnp.mean(store_energy)
        print('Estimated energy: ', gd_est[iteration]/d**2)
        if iteration % 100 == 0:
            if iteration > 0:
                np.save('gd_test.npy', gd_est)