#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Aug 21 10:06:48 2020

@author: Rob1
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import jax.numpy as jnp
from jax import grad
from jax import hessian
from jax.config import config
config.update("jax_enable_x64", True)

# pretty plots
font = {'family' : 'sans-serif',
        'weight' : 'regular',
        'size'   : 18}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 1.5

#%%

# ===============
# Exact solutions
# ===============

def tfi_exact(d, h):
    q = np.arange(1/d, 1, 2/d)
    ans = -np.mean(np.sqrt(1 + h**2 + 2*h*np.cos(np.pi*q)))
    return(ans)

def xxz_exact(d, delta):
    current = np.zeros(d//2)
    if delta < 1:
        eta = np.arccos(delta)
        def fixed_point(current, eta):
            d = 2*current.size
            quantum = np.arange(-d/4 + 1/2, d/4, 1)
            y = current[:, np.newaxis] - current[np.newaxis, :]
            y = 2*np.arctan(np.tanh(y)/np.tan(eta))
            y = np.sum(y, axis = 1)/d + 2*np.pi*quantum/d
            y = np.arctanh(np.tan(y/2)*np.tan(eta/2))
            return(y)
        for j in range(1000):
            current = np.copy(fixed_point(current, eta))
        ans = np.sum(-np.sin(eta)**2/(np.cosh(2*current)-np.cos(eta)))
        ans = 4*ans/d + delta    
    elif delta == 1:
        def fixed_point(current):
            d = 2*current.size
            quantum = np.arange(-d/4 + 1/2, d/4, 1)
            y = current[:, np.newaxis] - current[np.newaxis, :]
            y = 2*np.arctan(y)
            y = np.sum(y, axis = 1)/d + 2*np.pi*quantum/d
            y = np.tan(y/2)/2
            return(y)
        for j in range(1000):
            current = np.copy(fixed_point(current))
        ans = np.sum(-2/(4*current**2 + 1))
        ans = 4*ans/d + delta
    else:
        eta = jnp.arccosh(delta)
        quantum = jnp.arange(-d/4 + 1/2, d/4, 1)
        def jax_fp(current, eta, quantum):
            x = 2*jnp.arctan(jnp.tan(current)/jnp.tanh(eta/2))
            x = x + 2*jnp.pi*jnp.floor(current/jnp.pi + 1/2)
            diff = current[:, jnp.newaxis] - current[jnp.newaxis, :]
            y = 2*jnp.arctan(jnp.tan(diff)/jnp.tanh(eta))
            y = y + 2*jnp.pi*jnp.floor(diff/jnp.pi + 1/2)
            y = jnp.sum(y, axis = 1)/d + 2*jnp.pi*quantum/d
            z = jnp.sum((x-y)**2)
            return(z)
        jax_grad = grad(jax_fp, argnums = 0)    
        jax_hessian = hessian(jax_fp, argnums = 0)
        def wrapper_grad(current, eta, quantum):
            y = jax_grad(current, eta, quantum)
            return(np.array(y))
        def wrapper_hessian(current, eta, quantum):
            y = jax_hessian(current, eta, quantum)
            return(np.array(y))
        x0 = jnp.arctan(jnp.tan(jnp.pi*quantum/d)*jnp.tanh(eta/2))
        result = scipy.optimize.minimize(jax_fp, x0, args = (eta, quantum),
                                         method = 'trust-exact', jac = wrapper_grad,
                                         hess = wrapper_hessian)
        ans = np.sum(-np.sinh(eta)**2/(np.cosh(eta) - np.cos(2*result.x)))
        ans = 4*ans/d + delta
    return(ans)

#%%

# ========
# Table II
# ========

d = 200
folder = 'tfi_200x1/'
for name in ['0.5', '1.0', '1.5']:
    print('h = ', name)
    ans = tfi_exact(d, np.float(name))
    # SR
    sr = np.load(folder + 'tfi_' + name + '/sr_test.npy')
    sr_accuracy = np.abs(np.mean(sr/d)/ans - 1)
    print('SR rel. error: ', sr_accuracy)
    # NRGN
    nrgn = np.load(folder + 'tfi_' + name + '/nrgn_test.npy')
    nrgn_accuracy = np.abs(np.mean(nrgn/d)/ans - 1)
    print('RGN rel. error: ', nrgn_accuracy)
    print('Error ratio: ', sr_accuracy/nrgn_accuracy, '\n')

d = 100
folder = 'xxz_100x1/'
name = '0.5'
print('Delta = ', name)
ans = xxz_exact(d, np.float(name))
# NRGN
nrgn = np.load(folder + 'xxz_' + name + '/nrgn_test.npy')
nrgn_accuracy = np.abs(np.mean(nrgn/d)/ans - 1)
print('RGN rel. error: ', nrgn_accuracy)
# SR
sr = np.load(folder + 'xxz_' + name + '/sr_test.npy')
sr_accuracy = np.abs(np.mean(sr/d)/ans - 1)
print('SR rel. error: ', sr_accuracy)
print('Error ratio: ', sr_accuracy/nrgn_accuracy, '\n')

d = 100
folder = 'xxz_100x1_temper/'
for name in ['1.0', '1.5']:
    print('Delta = ', name)
    ans = xxz_exact(d, np.float(name))
    # NRGN
    nrgn = np.load(folder + 'xxz_' + name + '/nrgn_test.npy')
    nrgn_accuracy = np.abs(np.mean(nrgn/d)/ans - 1)
    print('RGN rel. error: ', nrgn_accuracy)
    # SR
    sr = np.load(folder + 'xxz_' + name + '/sr_test.npy')
    sr_accuracy = np.abs(np.mean(sr/d)/ans - 1)
    print('SR rel. error: ', sr_accuracy)
    print('Error ratio: ', sr_accuracy/nrgn_accuracy, '\n')
    
#%%

# =========
# Table III
# =========

d = 20
folder = 'tfi_20x20/'
for name in ['2.0', '3.0', '4.0']:
    print('h = ', name)
    # SR
    sr = np.load(folder + 'tfi_' + name + '/sr_test.npy')
    print('SR: ', np.real(np.mean(sr/d**2)))
    print('SR sigma: ', np.sqrt(np.var(sr/d**2)))
    # RGN
    nrgn = np.load(folder + 'tfi_' + name + '/nrgn_test.npy')
    print('RGN: ', np.real(np.mean(nrgn/d**2)))
    print('RGN sigma: ', np.sqrt(np.var(nrgn/d**2)), '\n')

#%%

# ========
# Figure I
# ========

# choose what to plot
d = 200
folder = 'tfi_200x1/'
name = '1.5'
iterations = 1001
scale = 1e-5

# plot results
ans = tfi_exact(d, np.float(name))
nrgn = np.real(np.load(folder + 'tfi_' + name + '/nrgn_est.npy'))
fig = plt.figure(figsize = (7, 4.25))
ax = fig.gca()
ax.plot(nrgn/d, color = '#FE6100', linewidth = 1.5, label = 'VMC estimate')
ax.axhline(ans, linestyle = '--', color = 'black', linewidth = 1.5, label = 'Ground-state energy')

# make it pretty
ax.set_xlim([0, iterations])
ax.set_ylim([ans - scale/4, ans + 3*scale/4])
ax.ticklabel_format(useOffset = False)
ax.tick_params(width = 1.5, which = 'both')
ax.set_xlabel('Iteration', font)
handles, labels = ax.get_legend_handles_labels()
order = [0, 1]
leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'upper right')
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.tight_layout()
fig.savefig('fig1.pdf', bbox_inches = 'tight', dpi = 200)

#%%

# =========
# Figure II
# =========

# choose what to plot
d = 10
folder = 'deterministic/'
ymin = 1e-10
ymax = 1
iterations = 1001
        
# plot results
fig, ax = plt.subplots(1, 3, figsize = (10, 5), sharey = True)
ans = tfi_exact(d, 0.5)
gd = np.load(folder + 'tfi_0.5/gd_exact.npy')
sr = np.load(folder + 'tfi_0.5/sr_exact.npy')
nrgn = np.load(folder + 'tfi_0.5/nrgn_exact.npy')
ax[0].semilogy(1 - gd/(d*ans), color = '#648FFF', linewidth = 1.5, label = 'GD')
ax[0].semilogy(1 - sr/(d*ans), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[0].semilogy(1 - nrgn/(d*ans), color = '#FE6100', linewidth = 1.5, label = 'RGN')
ans = tfi_exact(d, 1.0)
gd = np.load(folder + 'tfi_1.0/gd_exact.npy')
sr = np.load(folder + 'tfi_1.0/sr_exact.npy')
nrgn = np.load(folder + 'tfi_1.0/nrgn_exact.npy')
ax[1].semilogy(1 - gd/(d*ans), color = '#648FFF', linewidth = 1.5, label = 'GD')
ax[1].semilogy(1 - sr/(d*ans), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[1].semilogy(1 - nrgn/(d*ans), color = '#FE6100', linewidth = 1.5, label = 'RGN')
ans = tfi_exact(d, 1.5)
gd = np.load(folder + 'tfi_1.5/gd_exact.npy')
sr = np.load(folder + 'tfi_1.5/sr_exact.npy')
nrgn = np.load(folder + 'tfi_1.5/nrgn_exact.npy')
ax[2].semilogy(1 - gd/(d*ans), color = '#648FFF', linewidth = 1.5, label = 'GD')
ax[2].semilogy(1 - sr/(d*ans), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[2].semilogy(1 - nrgn/(d*ans), color = '#FE6100', linewidth = 1.5, label = 'RGN')

# make it pretty
ax[0].set_xlim([0, iterations])
ax[0].set_ylim([ymin, ymax])
ax[0].tick_params(width = 1.5, which = 'both')
ax[0].set_xticks([0, 500, 1000])
ax[0].set_xlabel('Iteration', font)
ax[0].set_ylabel('Energy error', font)
ax[0].set_title('$h = 0.5$', font)
ax[1].set_xlim([0, iterations])
ax[1].set_ylim([ymin, ymax])
ax[1].tick_params(width = 1.5, which = 'both')
ax[1].set_xticks([0, 500, 1000])
ax[1].set_xlabel('Iteration', font)
ax[1].set_title('$h = 1.0$', font)
ax[2].set_xlim([0, iterations])
ax[2].set_xticks([0, 500, 1000])
ax[2].set_ylim([ymin, ymax])
ax[2].tick_params(width = 1.5, which = 'both')
ax[2].set_xlabel('Iteration', font)
ax[2].set_title('$h = 1.5$', font)
fig.tight_layout()
handles, labels = ax[2].get_legend_handles_labels()
order = [0, 1, 2]
leg = ax[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                   loc = 'upper center', bbox_to_anchor=(0.5, 1.28),
                   ncol = 4)
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('fig2.pdf', bbox_inches = 'tight', dpi = 200)

#%%

#===========
# Figure III
# ==========

# choose what to plot
d = 10
folder = 'deterministic/'
ymin = 1e-6
ymax = 10
iterations = 1001

# plot results
fig, ax = plt.subplots(1, 3, figsize = (10, 4.5), sharey = True)
dist = np.sqrt(np.load(folder + 'tfi_0.5/nrgn_hessian.npy'))
ax[0].semilogy(dist, color = '#FE6100', linewidth = 1.5, label = 'Natural RGN')
dist = np.sqrt(np.load(folder + 'tfi_1.0/nrgn_hessian.npy'))
ax[1].semilogy(dist, color = '#FE6100', linewidth = 1.5, label = 'Natural RGN')
dist = np.sqrt(np.load(folder + 'tfi_1.5/nrgn_hessian.npy'))
ax[2].semilogy(dist, color = '#FE6100', linewidth = 1.5, label = 'Natural RGN')

# make it pretty
ax[0].set_xlim([0, iterations])
ax[0].set_ylim([ymin, ymax])
ax[0].tick_params(width = 1.5, which = 'both')
ax[0].set_xlabel('Iteration', font)
ax[0].set_ylabel('Hessian error', font)
ax[0].set_title('$h = 0.5$', font)
ax[1].set_xlim([0, iterations])
ax[1].set_ylim([ymin, ymax])
ax[1].tick_params(width = 1.5, which = 'both')
ax[1].set_xlabel('Iteration', font)
ax[1].set_title('$h = 1.0$', font)
ax[2].set_xlim([0, iterations])
ax[2].set_ylim([ymin, ymax])
ax[2].tick_params(width = 1.5, which = 'both')
ax[2].set_xlabel('Iteration', font)
ax[2].set_title('$h = 1.5$', font)
fig.tight_layout()
fig.savefig('fig3.pdf', bbox_inches = 'tight', dpi = 200)

#%%

# =========
# Figure IV
# =========

# choose what to plot
d = 100
folder = 'xxz_100x1/'
ymin = 1e-8
ymax = 1
iterations = 1001
        
# plot results
fig, ax = plt.subplots(1, 3, figsize = (10, 5), sharey = True)
ans = xxz_exact(d, 0.5)
sr = np.load(folder + 'xxz_0.5/sr_est.npy')
nrgn = np.load(folder + 'xxz_0.5/nrgn_est.npy')
ax[0].semilogy(np.abs(1 - sr/(d*ans)), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[0].semilogy(np.abs(1 - nrgn/(d*ans)), color = '#FE6100', linewidth = 1.5, label = 'RGN')
ans = xxz_exact(d, 1.0)
sr = np.load(folder + 'xxz_1.0/sr_est.npy')
nrgn = np.load(folder + 'xxz_1.0/nrgn_est.npy')
ax[1].semilogy(np.abs(1 - sr/(d*ans)), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[1].semilogy(np.abs(1 - nrgn/(d*ans)), color = '#FE6100', linewidth = 1.5, label = 'RGN')
ans = xxz_exact(d, 1.5)
sr = np.load(folder + 'xxz_1.5/sr_est.npy')
nrgn = np.load(folder + 'xxz_1.5/nrgn_est.npy')
ax[2].semilogy(np.abs(1 - sr/(d*ans)), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[2].semilogy(np.abs(1 - nrgn/(d*ans)), color = '#FE6100', linewidth = 1.5, label = 'RGN')

# make it pretty
ax[0].set_xlim([0, iterations])
ax[0].set_ylim([ymin, ymax])
ax[0].tick_params(width = 1.5, which = 'both')
ax[0].set_xticks([0, 500, 1000])
ax[0].set_xlabel('Iteration', font)
ax[0].set_ylabel('Energy error', font)
ax[0].set_title('$\Delta = 0.5$', font)
ax[1].set_xlim([0, iterations])
ax[1].set_ylim([ymin, ymax])
ax[1].tick_params(width = 1.5, which = 'both')
ax[1].set_xticks([0, 500, 1000])
ax[1].set_xlabel('Iteration', font)
ax[1].set_title('$\Delta = 1.0$', font)
ax[2].set_xlim([0, iterations])
ax[2].set_xticks([0, 500, 1000])
ax[2].set_ylim([ymin, ymax])
ax[2].tick_params(width = 1.5, which = 'both')
ax[2].set_xlabel('Iteration', font)
ax[2].set_title('$\Delta = 1.5$', font)
fig.tight_layout()
handles, labels = ax[2].get_legend_handles_labels()
order = [0, 1]
leg = ax[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                   loc = 'upper center', bbox_to_anchor=(0.5, 1.28),
                   ncol = 4)
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('fig4.pdf', bbox_inches = 'tight', dpi = 200)

#%%

# ========
# Figure V
# ========

# choose what to plot
d = 100
folder = 'xxz_100x1_temper/'
ymin = 1e-8
ymax = 1
iterations = 1001
        
# plot results
fig, ax = plt.subplots(1, 2, figsize = (8, 5), sharey = True)
ans = xxz_exact(d, 1.0)
sr = np.load(folder + 'xxz_1.0/sr_est.npy')
nrgn = np.load(folder + 'xxz_1.0/nrgn_est.npy')
ax[0].semilogy(np.abs(1 - sr/(d*ans)), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[0].semilogy(np.abs(1 - nrgn/(d*ans)), color = '#FE6100', linewidth = 1.5, label = 'RGN')
ans = xxz_exact(d, 1.5)
sr = np.load(folder + 'xxz_1.5/sr_est.npy')
nrgn = np.load(folder + 'xxz_1.5/nrgn_est.npy')
ax[1].semilogy(np.abs(1 - sr/(d*ans)), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[1].semilogy(np.abs(1 - nrgn/(d*ans)), color = '#FE6100', linewidth = 1.5, label = 'RGN')

# make it pretty
ax[0].set_xlim([0, iterations])
ax[0].set_ylim([ymin, ymax])
ax[0].tick_params(width = 1.5, which = 'both')
ax[0].set_xticks([0, 500, 1000])
ax[0].set_xlabel('Iteration', font)
ax[0].set_ylabel('Energy error', font)
ax[0].set_title('$\Delta = 1.0$', font)
ax[1].set_xlim([0, iterations])
ax[1].set_xticks([0, 500, 1000])
ax[1].set_ylim([ymin, ymax])
ax[1].tick_params(width = 1.5, which = 'both')
ax[1].set_xlabel('Iteration', font)
ax[1].set_title('$\Delta = 1.5$', font)
fig.tight_layout()
handles, labels = ax[1].get_legend_handles_labels()
order = [0, 1]
leg = ax[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                   loc = 'upper center', bbox_to_anchor=(-.12, 1.28),
                   ncol = 4)
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('fig5.pdf', bbox_inches = 'tight', dpi = 200)

#%%

# ========
# Figure 6
# ========

# choose what to plot
d = 20
folder = 'tfi_20x20/'
iterations = 1001
scale = 2e-4
   
# plot results
fig, ax = plt.subplots(1, 3, figsize = (14, 5))
sr = np.real(np.load(folder + 'tfi_2.0/sr_est.npy'))
nrgn = np.real(np.load(folder + 'tfi_2.0/nrgn_est.npy'))
ans = np.real(np.mean(nrgn[-100:])/d**2)
ax[0].plot(sr/d**2, color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[0].plot(nrgn/d**2, color = '#FE6100', linewidth = 1.5, label = 'RGN')
ax[0].set_ylim([ans - scale/4, ans + 3*scale/4])
sr = np.real(np.load(folder + 'tfi_3.0/sr_est.npy'))
nrgn = np.real(np.load(folder + 'tfi_3.0/nrgn_est.npy'))
ans = np.real(np.mean(nrgn[-100:])/d**2)
ax[1].plot(sr/d**2, color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[1].plot(nrgn/d**2, color = '#FE6100', linewidth = 1.5, label = 'RGN')
ax[1].set_ylim([ans - scale/4, ans + 3*scale/4])
sr = np.real(np.load(folder + 'tfi_4.0/sr_est.npy'))
nrgn = np.real(np.load(folder + 'tfi_4.0/nrgn_est.npy'))
ans = np.real(np.mean(nrgn[-100:])/d**2)
ax[2].plot(sr/d**2, color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[2].plot(nrgn/d**2, color = '#FE6100', linewidth = 1.5, label = 'RGN')
ax[2].set_ylim([ans - scale/4, ans + 3*scale/4])

# make it pretty
ax[0].set_xlim([0, iterations])
ax[0].ticklabel_format(useOffset = False)
ax[0].tick_params(width = 1.5, which = 'both')
ax[0].set_xlabel('Iteration', font)
ax[0].set_ylabel('Estimated energy', font)
ax[0].set_title('$h = 2.0$', font)
ax[1].set_xlim([0, iterations])
ax[1].ticklabel_format(useOffset = False)
ax[1].tick_params(width = 1.5, which = 'both')
ax[1].set_xlabel('Iteration', font)
ax[1].set_title('$h = 3.0$', font)
ax[2].set_xlim([0, iterations])
ax[2].ticklabel_format(useOffset = False)
ax[2].tick_params(width = 1.5, which = 'both')
ax[2].set_xlabel('Iteration', font)
ax[2].set_title('$h = 4.0$', font)
fig.tight_layout()
handles, labels = ax[2].get_legend_handles_labels()
order = [0, 1]
leg = ax[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                   loc = 'upper center', bbox_to_anchor=(0.5, 1.28),
                   ncol = 4)
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('fig6.pdf', bbox_inches = 'tight', dpi = 200)

#%%

# ========
# Figure 7
# ========

# choose what to plot
d = 200
folder = 'tfi_200x1/'
ymin = 1e-9
ymax = 1
iterations = 1001
        
# plot results
fig, ax = plt.subplots(1, 3, figsize = (10, 5), sharey = True)
ans = tfi_exact(d, 0.5)
sr = np.load(folder + 'tfi_0.5/sr_est.npy')
nrgn = np.load(folder + 'tfi_0.5/nrgn_est.npy')
ax[0].semilogy(np.abs(1 - sr/(d*ans)), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[0].semilogy(np.abs(1 - nrgn/(d*ans)), color = '#FE6100', linewidth = 1.5, label = 'RGN')
ans = tfi_exact(d, 1.0)
sr = np.load(folder + 'tfi_1.0/sr_est.npy')
nrgn = np.load(folder + 'tfi_1.0/nrgn_est.npy')
ax[1].semilogy(np.abs(1 - sr/(d*ans)), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[1].semilogy(np.abs(1 - nrgn/(d*ans)), color = '#FE6100', linewidth = 1.5, label = 'RGN')
ans = tfi_exact(d, 1.5)
sr = np.load(folder + 'tfi_1.5/sr_est.npy')
nrgn = np.load(folder + 'tfi_1.5/nrgn_est.npy')
ax[2].semilogy(np.abs(1 - sr/(d*ans)), color = '#785EF0', linewidth = 1.5, label = 'Natural GD')
ax[2].semilogy(np.abs(1 - nrgn/(d*ans)), color = '#FE6100', linewidth = 1.5, label = 'RGN')

# make it pretty
ax[0].set_xlim([0, iterations])
ax[0].set_ylim([ymin, ymax])
ax[0].tick_params(width = 1.5, which = 'both')
ax[0].set_xticks([0, 500, 1000])
ax[0].set_xlabel('Iteration', font)
ax[0].set_ylabel('Energy error', font)
ax[0].set_title('$h = 0.5$', font)
ax[1].set_xlim([0, iterations])
ax[1].set_ylim([ymin, ymax])
ax[1].tick_params(width = 1.5, which = 'both')
ax[1].set_xticks([0, 500, 1000])
ax[1].set_xlabel('Iteration', font)
ax[1].set_title('$h = 1.0$', font)
ax[2].set_xlim([0, iterations])
ax[2].set_xticks([0, 500, 1000])
ax[2].set_ylim([ymin, ymax])
ax[2].tick_params(width = 1.5, which = 'both')
ax[2].set_xlabel('Iteration', font)
ax[2].set_title('$h = 1.5$', font)
fig.tight_layout()
handles, labels = ax[2].get_legend_handles_labels()
order = [0, 1]
leg = ax[1].legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                   loc = 'upper center', bbox_to_anchor=(0.5, 1.28),
                   ncol = 4)
leg.get_frame().set_linewidth(1.5)
leg.get_frame().set_edgecolor('black')
fig.savefig('fig7.pdf', bbox_inches = 'tight', dpi = 200)
