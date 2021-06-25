# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:31:32 2020

@author: Rob
"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import jax.numpy as jnp
import jax.ops
import jax.lax
import jax.random as random
from jax import jit
from jax import vmap
from jax import hessian
from jax.config import config
config.update("jax_enable_x64", True)

#%%

# ==========
# Parameters
# ==========

# system
d = 10 # dimensionality of spin chain
h = 1.5 # strength of transverse field
alpha = 5 # number of RBM features
iterations = 1001 # number of parameter updates
relax_time = 100 # relaxation time

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
gd = True
gd_min = .001
gd_max = .01

# initialization
key = random.PRNGKey(123)
key, key1, key2 = random.split(key, num = 3)
weights_save = .001*random.normal(key1, shape = (alpha*(d + 1),)) \
    + .001j*random.normal(key2, shape = (alpha*(d + 1),))

# configs
configs = jnp.arange(2**d)[:, jnp.newaxis] >> jnp.arange(d)[::-1] & 1
configs = configs.astype(jnp.bool_)

# spin-z component of H
diffs = configs ^ configs[:, (jnp.arange(d) + 1) % d]
spin_z = 2*jnp.sum(diffs, axis = 1) - d

# neighbor states
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
matrix = matrix.toarray()

# subsetting based on parity
subset1 = (2*jnp.sum(configs, axis = 1) + configs[:, 0] <= d)
subset1 = jnp.arange(2**d)[subset1]
subset2 = jnp.dot(configs[subset1, :], 2**jnp.arange(d)[::-1])
subset2 = 2**d - 1 - subset2
matrix = matrix[subset1[:, None], subset1] + matrix[subset1[:, None], subset2]
configs = configs[subset1, :]
    
# solution
q = np.arange(1/d, 1, 2/d)
soln = -np.mean(np.sqrt(1 + h**2 + 2*h*np.cos(np.pi*q)))

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

@jit
def rayleigh(weights):
    bias = jnp.reshape(weights[-alpha:], (alpha, 1))
    features = jnp.reshape(weights[:-alpha], (alpha, d))
    features2 = jnp.fft.fft(features)
    y = jnp.exp(jansatz(configs, features2, bias))
    result = jnp.real(jnp.vdot(y, jnp.dot(matrix, y))/jnp.vdot(y, y))
    return(result)

@jit
def objective(weights):
        weights = weights[:alpha*(d+1)] + 1j*weights[alpha*(d+1):]
        bias = jnp.reshape(weights[-alpha:], (alpha, 1))
        features = jnp.reshape(weights[:-alpha], (alpha, d))
        features2 = jnp.fft.fft(features)
        configs2 = jnp.fft.fft(configs, axis = 1)
        angles = features2[jnp.newaxis, :, :]*jnp.conj(configs2)[:, jnp.newaxis, :]
        angles = jnp.fft.ifft(angles, axis = -1) + bias[jnp.newaxis, :, :]
        y = jnp.exp(jnp.sum(jnp.log(jnp.cosh(angles)), axis = (1, 2)))
        result = jnp.real(jnp.vdot(y, jnp.dot(matrix, y))/jnp.vdot(y, y))
        return(result)
hess = jit(hessian(objective))

# rotation matrix
rotate = jnp.eye(alpha*(d+1))
rotate1 = jnp.column_stack((rotate, 1j*rotate))
rotate2 = jnp.column_stack((rotate, -1j*rotate))
rotate = jnp.row_stack((rotate1, rotate2))/2

#%%

# =====================
# MCMC training -- NRGN
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
    weights = jnp.array(weights_save)

    # storage
    weight_log = np.zeros((iterations, weights.size)) + 0j
    nrgn_exact = np.zeros(iterations)
    hessian_error = np.zeros(iterations)

    # update the weights
    guidance = np.inf
    for iteration in range(iterations):
        print(iteration)

        # process weights
        bias = jnp.reshape(weights[-alpha:], (alpha, 1))
        features = jnp.reshape(weights[:-alpha], (alpha, d))
        features2 = jnp.fft.fft(features)
        
        # normalized wavefunction and derivatives
        vals = jnp.exp(jansatz(configs, features2, bias))
        vals = vals / jnp.sum(jnp.abs(vals)**2)**.5
        g1, g2 = jgradient(configs, features2, bias)
        grads = jnp.column_stack((jnp.reshape(g1, (-1, alpha*d)), g2))
        grads = grads - jnp.dot(jnp.abs(vals)**2, grads)[jnp.newaxis, :]
        grads = grads*vals[:, jnp.newaxis]
        forces = jnp.dot(grads.conj().T, jnp.dot(matrix, vals))
        cov = jnp.matmul(grads.conj().T, grads)
        linear = jnp.matmul(grads.conj().T, jnp.matmul(matrix, grads))
        linear = linear - jnp.vdot(vals, jnp.dot(matrix, vals))*cov
        regular = linear + (cov + regs[iteration]*jnp.eye(weights.size))/epsilons[iteration]
        
        # check size
        reset = False
        move = -jnp.linalg.solve(regular, forces)
        while np.sum(np.abs(move)**2)**.5 > 2*guidance:
            reset = True
            epsilons[iteration] = epsilons[iteration] / 2
            regular = linear + (cov + regs[iteration]*jnp.eye(weights.size))/epsilons[iteration]
            move = -jnp.linalg.solve(regular, forces)
        weights = weights + move
        if reset:
            print('Resetting')
            epsilons[iteration:] = epsilons_reset[:-iteration]
            regs[iteration:] = regs_reset[:-iteration]
        if iteration > 0:
            guidance = np.sum(np.abs((weights - weight_log[iteration - 1]))**2)**.5        
        weight_log[iteration, :] = weights

        # error
        nrgn_exact[iteration] = rayleigh(weights)
        print('Energy error: ', nrgn_exact[iteration]/d - soln)
        weights2 = jnp.concatenate((weights.real, weights.imag))
        newton = hess(weights2)
        newton = jnp.matmul(jnp.matmul(rotate, newton), rotate.conj().T)
        error = jnp.sum(jnp.abs(newton[:weights.size, weights.size:])**2)
        hessian_error[iteration] = 2*error/jnp.sum(jnp.abs(newton)**2)
        print('Hessian error: ', hessian_error[iteration])

    # save the results
    np.save('nrgn_weights.npy', weight_log)
    np.save('nrgn_exact.npy', nrgn_exact)
    np.save('nrgn_hessian.npy', hessian_error)
    
    
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
    weights = jnp.array(weights_save)

    # storage
    weight_log = np.zeros((iterations, weights.size)) + 0j
    sr_exact = np.zeros(iterations)

    # update the weights
    guidance = np.inf
    for iteration in range(iterations):
        print(iteration)

        # process weights
        bias = jnp.reshape(weights[-alpha:], (alpha, 1))
        features = jnp.reshape(weights[:-alpha], (alpha, d))
        features2 = jnp.fft.fft(features)
        
        # normalized wavefunction and derivatives
        vals = jnp.exp(jansatz(configs, features2, bias))
        vals = vals / jnp.sum(jnp.abs(vals)**2)**.5
        g1, g2 = jgradient(configs, features2, bias)
        grads = jnp.column_stack((jnp.reshape(g1, (-1, alpha*d)), g2))
        grads = grads - jnp.dot(jnp.abs(vals)**2, grads)[jnp.newaxis, :]
        grads = grads*vals[:, jnp.newaxis]
        forces = jnp.dot(grads.conj().T, jnp.dot(matrix, vals))
        cov = jnp.matmul(grads.conj().T, grads)
        regular = (cov + sr_reg*jnp.eye(weights.size))/epsilons[iteration]

        # check size
        reset = False
        move = -jnp.linalg.solve(regular, forces)
        while np.sum(np.abs(move)**2)**.5 > 2*guidance:
            reset = True
            epsilons[iteration] = epsilons[iteration] / 2
            regular = (cov + sr_reg*jnp.eye(weights.size))/epsilons[iteration]
            move = -jnp.linalg.solve(regular, forces)
        weights = weights + move
        if reset:
            print('Resetting')
            epsilons[iteration:] = epsilons_reset[:-iteration]
        if iteration > 0:
            guidance = np.sum(np.abs((weights - weight_log[iteration - 1]))**2)**.5        
        weight_log[iteration, :] = weights
        
        # error
        sr_exact[iteration] = rayleigh(weights)
        print('Energy error: ', sr_exact[iteration]/d - soln)

    # save the results
    np.save('sr_weights.npy', weight_log)
    np.save('sr_exact.npy', sr_exact)
    
#%%

#====================
# Exact training - GD
# ===================

if gd:

    # starting condition
    epsilons = np.arange(iterations)/relax_time
    epsilons = gd_min*(gd_max/gd_min)**epsilons
    epsilons[epsilons > gd_max] = gd_max
    epsilons_reset = np.copy(epsilons)
    weights = jnp.array(weights_save)

    # storage
    weight_log = np.zeros((iterations, weights.size)) + 0j
    gd_exact = np.zeros(iterations)

    # update the weights 
    guidance = np.inf
    for iteration in range(iterations):
        print(iteration)

        # process weights
        bias = jnp.reshape(weights[-alpha:], (alpha, 1))
        features = jnp.reshape(weights[:-alpha], (alpha, d))
        features2 = jnp.fft.fft(features)
        
        # normalized wavefunction and derivatives
        vals = jnp.exp(jansatz(configs, features2, bias))
        vals = vals / jnp.sum(jnp.abs(vals)**2)**.5
        g1, g2 = jgradient(configs, features2, bias)
        grads = jnp.column_stack((jnp.reshape(g1, (-1, alpha*d)), g2))
        grads = grads - jnp.dot(jnp.abs(vals)**2, grads)[jnp.newaxis, :]
        grads = grads*vals[:, jnp.newaxis]
        forces = jnp.dot(grads.conj().T, jnp.dot(matrix, vals))

        # check size
        reset = False
        move = -epsilons[iteration]*forces
        while np.sum(np.abs(move)**2)**.5 > 2*guidance:
            reset = True
            epsilons[iteration] = epsilons[iteration] / 2
            move = -epsilons[iteration]*forces
        weights = weights + move
        if reset:
            print('Resetting')
            epsilons[iteration:] = epsilons[:-iteration]
        if iteration > 0:
            guidance = np.sum(np.abs((weights - weight_log[iteration - 1]))**2)**.5        
        weight_log[iteration, :] = weights
        
        # error
        gd_exact[iteration] = rayleigh(weights)
        print('Energy error: ', gd_exact[iteration]/d - soln)
            
    # save the results
    np.save('gd_weights.npy', weight_log)
    np.save('gd_exact.npy', gd_exact)