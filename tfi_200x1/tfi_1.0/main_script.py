# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:31:32 2020

@author: Rob
"""

import os
os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count=48"
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
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
d = 200 # dimensionality of spin chain
h = 1.0 # strength of transverse field
alpha = 5 # number of RBM features
iterations = 1001 # number of parameter updates
relax_time = 500 # relaxation time
parallel = 50 # number of parallel MC chains
T = 20 # number of steps per MC chain

# rgn
rgn = True
rgn_min = .001
rgn_max = 1000
reg_min = .001
reg_max = .1
# sr
sr = True
sr_min = .001
sr_max = .01
sr_reg = .001
# gd
gd = False
gd_min = .001
gd_max = .01

# random seeds
cores = jax.local_device_count()
print('Cores:', cores)
key = random.PRNGKey(123)
key, key1, key2, key3 = random.split(key, num = 4)
key_save = random.split(key, cores)

# random weights
weights_save = .001*random.normal(key1, shape = (alpha*(d + 1),)) \
    + .001j*random.normal(key2, shape = (alpha*(d + 1),))

# random states
states_save = random.choice(key3, jnp.array([True, False]), shape = (cores, parallel, d))
flips = (2*jnp.sum(states_save, axis = -1) + states_save[..., 0] > d)
states_save = states_save ^ jnp.expand_dims(flips, -1)

# =============
# True solution
# =============

if d <= 20:
    # list of configurations
    configs = jnp.arange(2**d)[:, jnp.newaxis] >> jnp.arange(d)[::-1] & 1
    configs = configs.astype(jnp.bool_)

    # spin-z component of H
    diffs = configs ^ configs[:, (jnp.arange(d) + 1) % d]
    spin_z = 2*jnp.sum(diffs, axis = 1) - d

    # spin-x component of H
    spin_x = jnp.zeros(configs.shape, dtype = jnp.int_)
    for i in range(d):
        new = jax.ops.index_update(configs, jax.ops.index[:, i], ~configs[:, i])
        new = jnp.dot(new, 2**jnp.arange(d)[::-1])
        spin_x = jax.ops.index_update(spin_x, jax.ops.index[:, i], new)
    i_vals = jnp.repeat(jnp.arange(2**d), d)
    j_vals = jnp.ravel(spin_x)

    # sparse Hamiltonian
    data = jnp.append(spin_z, jnp.repeat(-h, i_vals.size))
    i_vals = jnp.append(jnp.arange(2**d), i_vals)
    j_vals = jnp.append(jnp.arange(2**d), j_vals)
    matrix = scipy.sparse.coo_matrix((data, (i_vals, j_vals)))
    matrix = matrix.tocsr()
    
    # subsetting based on parity
    subset1 = (2*jnp.sum(configs, axis = 1) + configs[:, 0] <= d)
    subset1 = jnp.arange(2**d)[subset1]
    subset2 = jnp.dot(configs[subset1, :], 2**jnp.arange(d)[::-1])
    subset2 = 2**d - 1 - subset2
    matrix = matrix[subset1[:, None], subset1] + matrix[subset1[:, None], subset2]
    configs = configs[subset1, :]
    
    # sparse or dense eigensolve
    if d <= 10:
        soln = scipy.linalg.eigh(jnp.array(matrix.toarray()), eigvals = (0, 0))
    else:
        soln = scipy.sparse.linalg.eigsh(matrix, k = 1, which = 'SA')

    # rayleigh quotient
    def rayleigh(weights):
        bias = jnp.reshape(weights[-alpha:], (alpha, 1))
        features = jnp.reshape(weights[:-alpha], (alpha, d))
        features2 = jnp.fft.fft(features)
        y = jnp.exp(jansatz(configs, features2, bias))
        return(jnp.real(jnp.vdot(y, matrix.dot(y))/jnp.vdot(y, y)))

#%%

# =================
# Ansatz + gradient
# =================

@jit
def ansatz(state, features2, bias):
    state2 = jnp.fft.fft(state)
    angles = jnp.fft.ifft(features2*jnp.conj(state2))+ bias
    log_wave = jnp.sum(jnp.log(jnp.cosh(angles)))
    return(log_wave)
ansatz1 = vmap(ansatz, (0, None, None), 0)
jansatz = jit(ansatz1)

@jit
def gradient(state, features2, bias):
    state2 = jnp.fft.fft(state)
    angles = jnp.fft.ifft(features2*jnp.conj(state2))+ bias
    y = jnp.tanh(angles)
    grad_bias = jnp.sum(y, axis = -1)
    y2 = jnp.fft.fft(y)
    grad_features = jnp.fft.ifft(y2*state2)
    return(grad_features, grad_bias)
gradient1 = vmap(gradient, (0, None, None), (0, 0))
jgradient = jit(gradient1)

# ==============
# Local energies
# ==============

@jit
def e_locs(states, features2, bias):
    # gradients
    g1, g2 = jgradient(states, features2, bias)
    grads = jnp.column_stack((jnp.reshape(g1, (parallel, -1)), g2))
    # spin-z
    diffs = states ^ states[:, (jnp.arange(d) + 1) % d]
    energies = 2*jnp.sum(diffs, axis = -1) - d
    # neighbor states
    states2 = jnp.expand_dims(states, axis = -2)
    states2 = jnp.repeat(states2, d + 1, axis = -2)
    i0 = jnp.arange(d)
    states2 = jax.ops.index_update(states2, jax.ops.index[:, i0, i0], \
                                    ~states2[:, i0, i0])
    # flip neighbors
    flips = (2*jnp.sum(states2, axis = -1) + states2[..., 0] > d)
    states2 = states2 ^ jnp.expand_dims(flips, -1)
    # fast evaluation
    locs = jloc(states2, energies, features2, bias)
    return(grads, locs)

@jit
def loc(states, energy, features2, bias):
    log_waves = jansatz(states, features2, bias)
    wave_ratios = jnp.exp(log_waves[:-1] - log_waves[-1])
    local = energy - h*jnp.sum(wave_ratios)
    return(local)
loc1 = vmap(loc, (0, 0, None, None), 0)
jloc = jit(loc1)

@jit
def e_locs_more(states, features2, bias):
    # spin-z
    diffs = states ^ states[:, (jnp.arange(d) + 1) % d]
    energies = 2*jnp.sum(diffs, axis = -1) - d
    # neighbor states
    states2 = jnp.expand_dims(states, axis = -2)
    states2 = jnp.repeat(states2, d + 1, axis = -2)
    i0 = jnp.arange(d)
    states2 = jax.ops.index_update(states2, jax.ops.index[:, i0, i0], \
                                    ~states2[:, i0, i0])
    # flip neighbors
    flips = (2*jnp.sum(states2, axis = -1) + states2[..., 0] > d)
    states2 = states2 ^ jnp.expand_dims(flips, -1)
    # fast evaluation
    g1, g2, locs, h1, h2 = jmore(states2, energies, features2, bias)
    grads = jnp.column_stack((jnp.reshape(g1, (parallel, -1)), g2))
    hams = jnp.column_stack((jnp.reshape(h1, (parallel, -1)), h2))
    return(grads, locs, hams)

@jit
def more(states, energy, features2, bias):
    # local energy
    log_waves = jansatz(states, features2, bias)
    wave_ratios = jnp.exp(log_waves[:-1] - log_waves[-1])
    local = energy - h*jnp.sum(wave_ratios)
    # gradients
    g1, g2 = jgradient(states, features2, bias)
    # local gradient energy
    loc1 = jnp.tensordot(wave_ratios, g1[:-1, ...], axes = (0,0))
    loc2 = jnp.tensordot(wave_ratios, g2[:-1, ...], axes = (0,0))
    loc1 = energy*g1[-1, ...] - h*loc1
    loc2 = energy*g2[-1, ...] - h*loc2
    return(g1[-1, ...], g2[-1, ...], local, loc1, loc2)
more1 = vmap(more, (0, 0, None, None), (0, 0, 0, 0, 0))
jmore = jit(more1)

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
    (states, currents, key, _, _), (store_grad, store_energy) = \
        jax.lax.scan(sample_less, (states, currents, key, features2, bias), None, T)
    return(states, key, store_grad, store_energy)
parallel_data = pmap(get_data, in_axes = (0, 0, None), out_axes = (0, 0, 0, 0))

@jit
def get_more_data(states, key, weights):
    # initialize
    bias = jnp.reshape(weights[-alpha:], (alpha, 1))
    features = jnp.reshape(weights[:-alpha], (alpha, d))
    features2 = jnp.fft.fft(features)
    currents = jansatz(states, features2, bias)
    # generate data
    (states, currents, key, _, _), \
        (store_grad, store_energy, store_ham) = \
        jax.lax.scan(sample_more, (states, currents, key, features2, bias), None, T)
    return(states, key, store_grad, store_energy, store_ham)
parallel_more_data = pmap(get_more_data, in_axes = (0, 0, None), out_axes = (0, 0, 0, 0, 0))

@jit
def sample_more(inputs, i):
    (states, currents, key, features2, bias), _ = \
        jax.lax.scan(update, inputs, None, d)
    grads, energies, hams = e_locs_more(states, features2, bias)
    return (states, currents, key, features2, bias), (grads, energies, hams)

@jit
def sample_less(inputs, i):
    (states, currents, key, features2, bias), _ = \
        jax.lax.scan(update, inputs, None, d)
    grads, locs = e_locs(states, features2, bias)
    return (states, currents, key, features2, bias), (grads, locs)

@jit
def update(inputs, i):
    (states, currents, key, features2, bias) = inputs
    key, key1, key2, key3, key4 = random.split(key, num = 5)
    # update
    i0 = jnp.arange(parallel)
    i1 = random.choice(key1, d, shape = (parallel,))
    perturbs = jax.ops.index_update(states, jax.ops.index[i0, i1],
                                    ~states[i0, i1])    
    # flip spins
    flips = (2*jnp.sum(perturbs, axis = -1) + perturbs[..., 0] > d)
    perturbs = perturbs ^ jnp.expand_dims(flips, -1)
    # accept or reject moves
    futures = jnp.real(jansatz(perturbs, features2, bias))
    accept = random.exponential(key2, shape = (parallel,))
    accept = (futures - currents > -.5*accept)
    # update information
    states = (states & ~accept[:, jnp.newaxis]) | (perturbs & accept[:, jnp.newaxis])
    currents = currents * ~accept + futures * accept
    return (states, currents, key, features2, bias), None

# ======================
# Multicore optimization
# ======================

def gd_multi(states, weights, key, epsilon, guidance):
    
    # get data
    (states, key, store_grad, store_energy) = parallel_data(states, key, weights)
    store_grad = jnp.reshape(store_grad, (-1, alpha*(d + 1)))
    store_energy = jnp.reshape(store_energy, (-1,))

    # form matrices
    ave = jnp.mean(store_energy)
    store_energy = store_energy - ave
    var = jnp.mean(np.abs(store_energy)**2)
    max_dev = jnp.max(jnp.abs(store_energy))/var**.5
    store_grad = store_grad - jnp.mean(store_grad, axis = 0)
    forces = jnp.dot(store_grad.conj().T, store_energy)/store_energy.size

    # check size
    reset = False
    move = -epsilon*forces
    while np.sum(np.abs(move)**2)**.5 > 2*guidance:
        reset = True
        epsilon = epsilon / 2
        move = -epsilon*forces
        
    # make a move
    weights = weights + move
    return(states, weights, key, ave, var, max_dev, reset)

def sr_multi(states, weights, key, epsilon, guidance):
    
    # get data
    (states, key, store_grad, store_energy) = parallel_data(states, key, weights)
    store_grad = jnp.reshape(store_grad, (-1, alpha*(d + 1)))
    store_energy = jnp.reshape(store_energy, (-1,))

    # form matrices
    ave = jnp.mean(store_energy)
    store_energy = store_energy - ave
    var = jnp.mean(np.abs(store_energy)**2)
    max_dev = jnp.max(jnp.abs(store_energy))/var**.5
    store_grad = store_grad - jnp.mean(store_grad, axis = 0)
    forces = jnp.dot(store_grad.conj().T, store_energy)/store_energy.size
    cov = jnp.matmul(store_grad.conj().T, store_grad)/store_energy.size
    regular = (cov + sr_reg*jnp.eye(weights.size))/epsilon

    # check size
    reset = False
    move = -jnp.linalg.solve(regular, forces)
    while np.sum(np.abs(move)**2)**.5 > 2*guidance:
        reset = True
        epsilon = epsilon / 2
        regular = (cov + sr_reg*jnp.eye(weights.size))/epsilon
        move = -jnp.linalg.solve(regular, forces)

    # make a move
    weights = weights + move
    return(states, weights, key, ave, var, max_dev, reset)

def rgn_multi(states, weights, key, epsilon, reg, guidance):

    # get data
    (states, key, store_grad, store_energy, store_ham) = \
        parallel_more_data(states, key, weights)
    store_grad = jnp.reshape(store_grad, (-1, alpha*(d + 1)))
    store_energy = jnp.reshape(store_energy, (-1,))
    store_ham = jnp.reshape(store_ham, (-1, alpha*(d + 1)))

    # form matrices
    ave = jnp.mean(store_energy)
    store_energy = store_energy - ave
    var = jnp.mean(np.abs(store_energy)**2)
    max_dev = jnp.max(jnp.abs(store_energy))/var**.5
    ave_grad = jnp.mean(store_grad, axis = 0)
    store_grad = store_grad - ave_grad
    store_ham = store_ham - jnp.mean(store_ham, axis = 0)
    forces = jnp.dot(store_grad.conj().T, store_energy)/store_energy.size
    cov = jnp.matmul(store_grad.conj().T, store_grad)/store_energy.size
    linear = jnp.dot(store_grad.conj().T, store_ham)/store_energy.size
    linear = linear - jnp.outer(forces, ave_grad) - ave*cov
    regular = linear + (cov + reg*jnp.eye(weights.size))/epsilon

    # check size
    reset = False
    move = -jnp.linalg.solve(regular, forces)
    while np.sum(np.abs(move)**2)**.5 > 2*guidance:
        reset = True
        epsilon = epsilon / 2
        regular = linear + (cov + reg*jnp.eye(weights.size))/epsilon
        move = -jnp.linalg.solve(regular, forces)

    # make a move
    weights = weights + move
    return(states, weights, key, ave, var, max_dev, reset)

#%%

# =====================
# MCMC training -- RGN
# =====================

if rgn:

    # starting condition
    epsilons = np.arange(iterations)/relax_time
    epsilons = rgn_min*(rgn_max/rgn_min)**epsilons
    epsilons[epsilons > rgn_max] = rgn_max
    epsilons_reset = np.copy(epsilons)
    regs = np.arange(iterations)/relax_time
    regs = reg_min*(reg_max/reg_min)**regs
    regs[regs > reg_max] = reg_max
    regs_reset = np.copy(regs)
    key = jnp.array(key_save)
    weights = jnp.array(weights_save)
    states = jnp.array(states_save)

    # storage
    weight_log = np.zeros((iterations, weights.size)) + 0j
    rgn_est = np.zeros(iterations) + 0j
    rgn_var = np.zeros(iterations)
    rgn_dev = np.zeros(iterations)
    if d <= 20:
        rgn_exact = np.zeros(iterations)

    # update the weights
    guidance = np.inf
    for iteration in range(iterations):
        print(iteration)
        (states, weights, key, rgn_est[iteration], 
         rgn_var[iteration], rgn_dev[iteration], reset) = \
            rgn_multi(states, weights, key, 
                      epsilons[iteration], regs[iteration], guidance)
        if reset:
            print('Resetting')
            np.save('nrgn_error_' + str(iteration) + '.npy', states)
            epsilons[iteration:] = epsilons_reset[:-iteration]
            regs[iteration:] = regs_reset[:-iteration]
        if iteration > 0:
            guidance = np.sum(np.abs((weights - weight_log[iteration - 1, :]))**2)**.5
        weight_log[iteration, :] = weights
        
        # report progress
        print('Estimated energy: ', rgn_est[iteration]/d)
        print('Energy variance: ', rgn_var[iteration]/d**2)
        print('Maximum deviation: ', rgn_dev[iteration])
        # compare to truth
        if d <= 20:
            rgn_exact[iteration] = rayleigh(weights)
            print('Exact distance: ', (rgn_exact[iteration] - soln[0][0])/d)

        # save the results
        if iteration % 100 == 0:
            if iteration > 0:
                np.save('nrgn_save_' + str(iteration) + '.npy', states)
                np.save('nrgn_weights.npy', weight_log)
                np.save('nrgn_est.npy', rgn_est)
                np.save('nrgn_var.npy', rgn_var)
                np.save('nrgn_dev.npy', rgn_dev)
                if d <= 20:
                    np.save('nrgn_exact.npy', rgn_exact)
                    
#%%

#===================
# MCMC training - SR
# ==================

if sr:

    # starting condition
    epsilons = np.arange(iterations)/relax_time
    epsilons = sr_min*(sr_max/sr_min)**epsilons
    epsilons[epsilons > sr_max] = sr_max
    epsilons_reset = np.copy(epsilons)
    key = jnp.array(key_save)
    weights = jnp.array(weights_save)
    states = jnp.array(states_save)

    # storage
    weight_log = np.zeros((iterations, weights.size)) + 0j
    sr_est = np.zeros(iterations) + 0j
    sr_var = np.zeros(iterations)
    sr_dev = np.zeros(iterations)
    if d <= 20:
        sr_exact = np.zeros(iterations)

    # update the weights
    guidance = np.inf
    for iteration in range(iterations):
        print(iteration)
        (states, weights, key, sr_est[iteration], 
         sr_var[iteration], sr_dev[iteration], reset) = \
            sr_multi(states, weights, key, epsilons[iteration], guidance)
        if reset:
            print('Resetting')
            np.save('sr_error_' + str(iteration) + '.npy', states)
            epsilons[iteration:] = epsilons_reset[:-iteration]
        if iteration > 0:
            guidance = np.sum(np.abs((weights - weight_log[iteration - 1, :]))**2)**.5        
        weight_log[iteration, :] = weights
        
        # report progress
        print('Estimated energy: ', sr_est[iteration]/d)
        print('Energy variance: ', sr_var[iteration]/d**2)
        print('Maximum deviation: ', sr_dev[iteration])
        if d <= 20:
            sr_exact[iteration] = rayleigh(weights)
            print('Exact distance: ', (sr_exact[iteration] - soln[0][0])/d)

        # save the results
        if iteration % 100 == 0:
            if iteration > 0:
                np.save('sr_save_' + str(iteration) + '.npy', states)
                np.save('sr_weights.npy', weight_log)
                np.save('sr_est.npy', sr_est)
                np.save('sr_var.npy', sr_var)
                np.save('sr_dev.npy', sr_dev)
                if d <= 20:
                    np.save('sr_exact.npy', sr_exact)
                    
#%%

#===================
# MCMC training - GD
# ==================

if gd:

    # starting condition
    epsilons = np.arange(iterations)/relax_time
    epsilons = gd_min*(gd_max/gd_min)**epsilons
    epsilons[epsilons > gd_max] = gd_max
    epsilons_reset = np.copy(epsilons)
    key = jnp.array(key_save)
    weights = jnp.array(weights_save)
    states = jnp.array(states_save)

    # storage
    weight_log = np.zeros((iterations, weights.size))
    gd_est = np.zeros(iterations) + 0j
    gd_var = np.zeros(iterations)
    gd_dev = np.zeros(iterations)
    if d <= 20:
        gd_exact = np.zeros(iterations)

    # update the weights 
    guidance = np.inf
    for iteration in range(iterations):
        print(iteration)
        (states, weights, key, gd_est[iteration], 
         gd_var[iteration], gd_dev[iteration], reset) = \
            gd_multi(states, weights, key, epsilons[iteration], guidance)
        if reset:
            print('Resetting')
            np.save('gd_error_' + str(iteration) + '.npy', states)
            epsilons[iteration:] = epsilons_reset[:-iteration]
        if iteration > 0:
            guidance = np.sum(np.abs((weights - weight_log[iteration - 1, :]))**2)**.5        
        weight_log[iteration, :] = weights

        # report progress
        print('Estimated energy: ', gd_est[iteration]/d)
        print('Energy variance: ', gd_var[iteration]/d**2)
        print('Maximum deviation: ', gd_dev[iteration])
        if d <= 20:
            gd_exact[iteration] = rayleigh(weights)
            print('Exact error: ', (gd_exact[iteration] - soln[0][0])/d)

        # save the results
        if iteration % 100 == 0:
            if iteration > 0:
                np.save('gd_save_' + str(iteration) + '.npy', states)
                np.save('gd_weights.npy', weight_log)
                np.save('gd_est.npy', gd_est)
                np.save('gd_var.npy', gd_var)
                np.save('gd_dev.npy', gd_dev)
                if d <= 20:
                    np.save('gd_exact.npy', gd_exact)
