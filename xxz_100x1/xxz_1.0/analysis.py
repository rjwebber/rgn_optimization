import os
import numpy as np
import matplotlib.pyplot as plt
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
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=4"
config.update("jax_enable_x64", True)

# system
d = 100  # dimensionality of spin chain
h = 1.0  # strength of z interactions
alpha = 5  # number of RBM features
parallel = 50  # number of parallel MC chains
T = 20  # number of steps per MC chain

# random seeds
cores = jax.local_device_count()
print('Cores:', cores)


@jit
def ansatz(state, features2, bias):
    state2 = jnp.fft.fft(state)
    angles = jnp.fft.ifft(features2 * jnp.conj(state2)) + bias
    log_wave = jnp.sum(jnp.log(jnp.cosh(angles)))
    return(log_wave)


ansatz1 = vmap(ansatz, (0, None, None), 0)
jansatz = jit(ansatz1)


@jit
def loc(states, sames, energy, features2, bias):
    log_waves = jansatz(states, features2, bias)
    wave_ratios = jnp.exp(log_waves[:-1] - log_waves[-1])
    wave_ratios = wave_ratios * sames
    local = energy - 2 * jnp.sum(wave_ratios)
    return(local)


loc1 = vmap(loc, (0, 0, 0, None, None), 0)
jloc = jit(loc1)


@jit
def e_locs(states, features2, bias):
    # spin-z
    sames = ~(states ^ states[:, (jnp.arange(d) + 1) % d])
    energies = h * (d - 2 * jnp.sum(sames, axis=-1))
    # neighbor states
    states2 = jnp.expand_dims(states, axis=-2)
    states2 = jnp.repeat(states2, d + 1, axis=-2)
    i0 = jnp.arange(d)
    states2 = jax.ops.index_update(states2, jax.ops.index[:, i0, i0],
                                   ~states2[:, i0, i0])
    states2 = jax.ops.index_update(states2, jax.ops.index[:, i0, (i0 + 1) % d],
                                   ~states2[:, i0, (i0 + 1) % d])
    # flip neighbors
    flips = (2 * jnp.sum(states2, axis=-1) + states2[..., 0] > d)
    states2 = states2 ^ jnp.expand_dims(flips, -1)
    # fast evaluation
    locs = jloc(states2, sames, energies, features2, bias)
    return(locs)


#%%

# pretty plots
font = {'family': 'sans-serif',
        'weight': 'regular',
        'size': 13}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 1.5

weights = np.load('rgn_weights.npy')
ests = np.load('rgn_est.npy')
bins_list = np.linspace(-.98, .98, 50)
bins_list = np.concatenate(([-1.], bins_list, [1.]))
fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

# states before
t = 400
states = np.load('rgn_save_' + str(t) + '.npy')
states = states.reshape(-1, d)
magnets = np.sum(states ^ states[:, (np.arange(d) + 1) % d], -1)
magnets = 1 - magnets * 2 / d
ax[0].hist(magnets, bins=bins_list, color='#FE6100', edgecolor='black')
ax[0].set_xlim([-1, 1])
ax[0].set_xlabel('Magnetization')
ax[0].set_ylim([0, 590])
ax[0].set_ylabel('Number of samples')
ax[0].text(.9, 540, r'$-14 < \log |\psi| < 6$', font, ha='right')
ax[0].text(.9, 490, r'$-1.79 < E_L < -1.76$', font, ha='right')
ax[0].text(-.9, 90, r'$409 < \log |\psi| < 410$', font, ha='left')
ax[0].text(-.9, 40, r'$0.94 < E_L < 0.95$', font, ha='left')
ax[0].tick_params(width=1.5, which='both')
ax[0].set_title(r'Early spike: $t = 400$', font)

# # energies
# bias = jnp.reshape(weights[t, -alpha:], (alpha, 1))
# features = jnp.reshape(weights[t, :-alpha], (alpha, d))
# features2 = jnp.fft.fft(features)
# energies = np.real(e_locs(states, features2, bias)) / d
# normal_energies = energies[magnets <= 0]
# print(normal_energies.min(), normal_energies.max())
# weird_energies = energies[magnets > 0]
# print(weird_energies.min(), weird_energies.max())
# # densities
# ansatz = np.real(jansatz(states, features2, bias))
# normal_ansatz = ansatz[magnets <= 0]
# print(normal_ansatz.min(), normal_ansatz.max())
# weird_ansatz = ansatz[magnets > 0]
# print(weird_ansatz.min(), weird_ansatz.max())

# states before
t = 800
states = np.load('rgn_save_' + str(t) + '.npy')
states = states.reshape(-1, d)
magnets = np.sum(states ^ states[:, (np.arange(d) + 1) % d], -1)
magnets = 1 - magnets * 2 / d
ax[1].hist(magnets, bins=bins_list, color='#FE6100', edgecolor='black')
ax[1].set_xlim([-1, 1])
ax[1].set_xlabel('Magnetization')
ax[1].text(.9, 540, r'$-13 < \log |\psi| < 7$', font, ha='right')
ax[1].text(.9, 490, r'$-1.79 < E_L < -1.76$', font, ha='right')
ax[1].text(-.9, 90, r'$223 < \log |\psi| < 228$', font, ha='left')
ax[1].text(-.9, 40, r'$0.50 < E_L < 0.74$', font, ha='left')
ax[1].tick_params(width=1.5, which='both')
ax[1].set_title(r'Mid spike: $t = 800$', font)

# states before
t = 1200
states = np.load('rgn_save_' + str(t) + '.npy')
states = states.reshape(-1, d)
magnets = np.sum(states ^ states[:, (np.arange(d) + 1) % d], -1)
magnets = 1 - magnets * 2 / d
ax[2].hist(magnets, bins=bins_list, color='#FE6100', edgecolor='black')
ax[2].set_xlim([-1, 1])
ax[2].set_xlabel('Magnetization')
ax[2].text(.9, 540, r'$14 < \log |\psi| < 32$', font, ha='right')
ax[2].text(.9, 490, r'$-1.78 < E_L < -1.76$', font, ha='right')
ax[2].text(-.9, 90, r'$2 < \log |\psi| < 15$', font, ha='left')
ax[2].text(-.9, 40, r'$0.23 < E_L < 0.71$', font, ha='left')
ax[2].tick_params(width=1.5, which='both')
ax[2].set_title('Late spike: $t = 1200$', font)

fig.tight_layout()
fig.savefig('fig_analysis.pdf', bbox_inches='tight', dpi=200)

#%%

# pretty plots
font = {'family': 'sans-serif',
        'weight': 'regular',
        'size': 12}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 1.5

weights = np.load('rgn_weights.npy')
ests = np.load('rgn_est.npy')
bins_list = np.linspace(-.98, .98, 50)
bins_list = np.concatenate(([-1.], bins_list, [1.]))
fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

# states before
t = 300
states = np.load('rgn_save_' + str(t) + '.npy')[:, :50, :]
states = states.reshape(-1, d)
magnets = np.sum(states ^ states[:, (np.arange(d) + 1) % d], -1)
magnets = 1 - magnets * 2 / d
ax[0].hist(magnets, bins=bins_list, color='#FE6100', edgecolor='black')
ax[0].set_xlim([-1, 1])
ax[0].set_xlabel('Magnetization')
ax[0].set_ylim([0, 590])
#ax[0].set_yscale('log')
ax[0].set_ylabel('Number of samples')
ax[0].text(.9, 540, r'$-25 < \log |\psi| < -6$', font, ha='right')
ax[0].text(.9, 490, r'$-1.79 < E_L < -1.76$', font, ha='right')
ax[0].text(-.9, 90, r'$34 < \log |\psi| < 37$', font, ha='left')
ax[0].text(-.9, 40, r'$0.69 < E_L < 0.70$', font, ha='left')
ax[0].tick_params(width=1.5, which='both')
ax[0].set_title(r'Early spike: $t = 300$', font)

# states before
t = 400
states = np.load('rgn_save_' + str(t) + '.npy')[:, :50, :]
states = states.reshape(-1, d)
magnets = np.sum(states ^ states[:, (np.arange(d) + 1) % d], -1)
magnets = 1 - magnets * 2 / d
ax[1].hist(magnets, bins=bins_list, color='#FE6100', edgecolor='black')
ax[1].set_xlim([-1, 1])
ax[1].set_xlabel('Magnetization')
ax[1].text(.9, 540, r'$23 < \log |\psi| < 40$', font, ha='right')
ax[1].text(.9, 490, r'$-1.78 < E_L < -1.76$', font, ha='right')
ax[1].text(-.9, 90, r'$43 < \log |\psi| < 46$', font, ha='left')
ax[1].text(-.9, 40, r'$0.55 < E_L < 0.57$', font, ha='left')
ax[1].tick_params(width=1.5, which='both')
ax[1].set_title(r'Mid spike: $t = 400$', font)

# states before
t = 500
states = np.load('rgn_save_' + str(t) + '.npy')[:, :50, :]
states = states.reshape(-1, d)
magnets = np.sum(states ^ states[:, (np.arange(d) + 1) % d], -1)
magnets = 1 - magnets * 2 / d
ax[2].hist(magnets, bins=bins_list, color='#FE6100', edgecolor='black')
ax[2].set_xlim([-1, 1])
ax[2].set_xlabel('Magnetization')
ax[2].text(.9, 540, r'$53 < \log |\psi| < 60$', font, ha='right')
ax[2].text(.9, 490, r'$-1.78 < E_L < -1.77$', font, ha='right')
ax[2].text(-.9, 90, r'$39 < \log |\psi| < 56$', font, ha='left')
ax[2].text(-.9, 40, r'$-0.02 < E_L < 0.64$', font, ha='left')
ax[2].tick_params(width=1.5, which='both')
ax[2].set_title('Late spike: $t = 500$', font)

# # energies
# bias = jnp.reshape(weights[t, -alpha:], (alpha, 1))
# features = jnp.reshape(weights[t, :-alpha], (alpha, d))
# features2 = jnp.fft.fft(features)
# energies = np.real(e_locs(states, features2, bias)) / d
# normal_energies = energies[magnets <= 0]
# print(normal_energies.min(), normal_energies.max())
# weird_energies = energies[magnets > 0]
# print(weird_energies.min(), weird_energies.max())
# # densities
# ansatz = np.real(jansatz(states, features2, bias))
# normal_ansatz = ansatz[magnets <= 0]
# print(normal_ansatz.min(), normal_ansatz.max())
# weird_ansatz = ansatz[magnets > 0]
# print(weird_ansatz.min(), weird_ansatz.max())

fig.tight_layout()
fig.savefig('fig_analysis.pdf', bbox_inches='tight', dpi=200)