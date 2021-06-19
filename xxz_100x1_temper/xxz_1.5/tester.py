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
d = 100 # dimensionality of spin chain
h = 1.5 # strength of z interactions
alpha = 5 # number of RBM features
iterations = 101 # number of parameter updates
parallel = 50 # number of parallel MC chains
temps = 50 # number of temperatures
T = 20 # number of steps per MC chain

# rgn
rgn = True
# sr
sr = True
# gd
gd = False

# tempering
temps = temps - 1
parallel = temps*parallel
temper = jnp.repeat(np.arange(temps)/temps, parallel // temps)
part1 = jnp.arange(parallel)
part2 = part1[part1 % (2*parallel // temps) >= parallel // temps]
part1 = part1[part1 % (2*parallel // temps) < parallel // temps]
part1a = part1[(parallel // temps):]
part1b = part1[:-(parallel // temps)]
states_old = jnp.tile(jnp.array([True, False]), d//2)

# random seeds
cores = jax.local_device_count()
print('Cores:', cores)
key = random.PRNGKey(123)
key_save = random.split(key, cores)

#%%

# =================
# Ansatz + gradient
# =================

@jit
def ansatz(state, features2, bias):
    state2 = jnp.fft.fft(state)
    angles = jnp.fft.ifft(features2*jnp.conj(state2)) + bias
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
    sames = ~(states ^ states[:, (jnp.arange(d) + 1) % d])
    energies = h*(d - 2*jnp.sum(sames, axis = -1))
    # neighbor states
    states2 = jnp.expand_dims(states, axis = -2)
    states2 = jnp.repeat(states2, d + 1, axis = -2)
    i0 = jnp.arange(d)
    states2 = jax.ops.index_update(states2, jax.ops.index[:, i0, i0], \
                                    ~states2[:, i0, i0])
    states2 = jax.ops.index_update(states2, jax.ops.index[:, i0, (i0 + 1)%d], \
                                    ~states2[:, i0, (i0 + 1)%d])
    # flip neighbors
    flips = (2*jnp.sum(states2, axis = -1) + states2[..., 0] > d)
    states2 = states2 ^ jnp.expand_dims(flips, -1)
    # fast evaluation
    locs = jloc(states2, sames, energies, features2, bias)
    return(locs)

@jit
def loc(states, sames, energy, features2, bias):
    log_waves = jansatz(states, features2, bias)
    wave_ratios = jnp.exp(log_waves[:-1] - log_waves[-1])
    wave_ratios = wave_ratios*sames
    local = energy - 2*jnp.sum(wave_ratios)
    return(local)
loc1 = vmap(loc, (0, 0, 0, None, None), 0)
jloc = jit(loc1)

# ========
# Sampling
# ========

@jit
def get_data(states, key, weights):
    # initialize
    bias = jnp.reshape(weights[-alpha:], (alpha, 1))
    features = jnp.reshape(weights[:-alpha], (alpha, d))
    features2 = jnp.fft.fft(features)
    currents = jansatz(states, features2, bias)
    # generate data
    (states, currents, key, _, _), (store_energy) = \
        jax.lax.scan(sample_less, (states, currents, key, features2, bias), None, T)
    return(states, key, store_energy)
parallel_data = pmap(get_data, in_axes = (0, 0, None), out_axes = (0, 0, 0))

@jit
def sample_less(inputs, i):
    (states, currents, key, features2, bias), _ = \
        jax.lax.scan(update, inputs, None, d)
    locs = e_locs(states[:parallel//temps], features2, bias)
    return (states, currents, key, features2, bias), (locs)

@jit
def update(inputs, i):
    (states, currents, key, features2, bias) = inputs
    key, key1, key2, key3, key4, key5, key6, key7 = random.split(key, num = 8)
    # update
    i0 = jnp.arange(parallel)
    perturbs = jax.ops.index_update(states, jax.ops.index[:, ::2], ~states[:, ::2])
    ups = jnp.cumsum(perturbs, axis = 1)
    select = random.uniform(key1, shape = (parallel,))*(d // 2)
    i1 = jnp.argmax(jnp.greater(ups, select[:, jnp.newaxis]), axis = 1)
    downs = jnp.cumsum(~perturbs, axis = 1)
    select = random.uniform(key2, shape = (parallel,))*(d // 2)
    i2 = jnp.argmax(jnp.greater(downs, select[:, jnp.newaxis]), axis = 1)
    perturbs = jax.ops.index_update(states, jax.ops.index[i0, i1], ~states[i0, i1])
    perturbs = jax.ops.index_update(perturbs, jax.ops.index[i0, i2], ~states[i0, i2])
    # flip spins
    flips = (2*jnp.sum(perturbs, axis = -1) + perturbs[..., 0] > d)
    perturbs = perturbs ^ jnp.expand_dims(flips, -1)
    # accept or reject moves
    futures = jnp.real(jansatz(perturbs, features2, bias))
    accept = random.exponential(key3, shape = (parallel,))
    # accept = (futures - currents > -.5*accept)
    accept = (1-temper)*(futures - currents) > -.5*accept
    # update information
    states = (states & ~accept[:, jnp.newaxis]) | (perturbs & accept[:, jnp.newaxis])
    currents = currents * ~accept + futures * accept  
    # swap 0
    key4 = random.split(key4, num = parallel // temps)
    perturbs = vmap(random.permutation, in_axes = (0, None), out_axes = 0)(key4, states_old)
    perturbs = jax.ops.index_update(perturbs, jax.ops.index[:, ::2], ~perturbs[:, ::2])
    # flip spins
    flips = (2*jnp.sum(perturbs, axis = -1) + perturbs[..., 0] > d)
    perturbs = perturbs ^ jnp.expand_dims(flips, -1)
    # accept or reject moves
    futures = jnp.real(jansatz(perturbs, features2, bias))
    accept = random.exponential(key5, shape = (parallel // temps,))
    # accept = (futures - currents > -.5*accept)
    accept = (futures - currents[-(parallel // temps):])/temps > -.5*accept
    # update information
    perturbs = (states[-(parallel // temps):, :] & ~accept[:, jnp.newaxis]) | (perturbs & accept[:, jnp.newaxis])
    states = jax.ops.index_update(states, jax.ops.index[-(parallel // temps):, :], perturbs)
    futures = currents[-(parallel // temps):] * ~accept + futures * accept
    currents = jax.ops.index_update(currents, jax.ops.index[-(parallel // temps):], futures)
    # swap 1
    accept = random.exponential(key6, shape = part2.shape)
    diffs = currents[part1a] - currents[part2]
    accept = diffs > -.5*accept*temps
    currents = jax.ops.index_add(currents, jax.ops.index[part2], diffs*accept)
    currents = jax.ops.index_add(currents, jax.ops.index[part1a], -diffs*accept)
    accept = accept[:, jnp.newaxis]
    perturbs1 = states[part2, :] * ~accept + states[part1a, :] * accept
    perturbs2 = states[part2, :] * accept + states[part1a, :] * ~accept
    states = jax.ops.index_update(states, jax.ops.index[part2], perturbs1)
    states = jax.ops.index_update(states, jax.ops.index[part1a], perturbs2)
    # swap 2
    accept = random.exponential(key7, shape = part2.shape)
    diffs = currents[part2] - currents[part1b]
    accept = diffs > -.5*accept*temps
    currents = jax.ops.index_add(currents, jax.ops.index[part1b], diffs*accept)
    currents = jax.ops.index_add(currents, jax.ops.index[part2], -diffs*accept)
    accept = accept[:, jnp.newaxis]
    perturbs1 = states[part1b, :] * ~accept + states[part2, :] * accept
    perturbs2 = states[part1b, :] * accept + states[part2, :] * ~accept
    states = jax.ops.index_update(states, jax.ops.index[part1b], perturbs1)
    states = jax.ops.index_update(states, jax.ops.index[part2], perturbs2)    

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
        print('Estimated energy: ', rgn_est[iteration]/d)
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
        print('Estimated energy: ', sr_est[iteration]/d)
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
        print('Estimated energy: ', gd_est[iteration]/d)
        if iteration % 100 == 0:
            if iteration > 0:
                np.save('gd_test.npy', gd_est)