import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc
from matplotlib import animation

# pretty plots
palette = cc.colorwheel
cmap = plt.get_cmap('cet_colorwheel')
font = {'family': 'sans-serif',
        'weight': 'regular',
        'size': 18}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 1.5

# animation settings (not used)
# plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
# writer = animation.FFMpegWriter(bitrate=500, fps=8)

# %%

# =======
# Video I
# =======

# choose what to plot
d = 200
alpha = 5
folder = 'tfi_200x1/tfi_0.5/'
iterations = 1001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 199.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 50, 100, 150, 200]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 50, 100, 150, 200])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$200 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$200 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_200x1_h=0.5_rgn.mp4')

# %%

# ========
# Video II
# ========

# choose what to plot
d = 200
alpha = 5
folder = 'tfi_200x1/tfi_0.5/'
iterations = 1001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 199.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 50, 100, 150, 200]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 50, 100, 150, 200])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$200 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$200 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_200x1_h=0.5_sr.mp4')

# %%

# =========
# Video III
# =========

# choose what to plot
d = 200
alpha = 5
folder = 'tfi_200x1/tfi_1.0/'
iterations = 1001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 199.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 50, 100, 150, 200]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 50, 100, 150, 200])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$200 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$200 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_200x1_h=1.0_rgn.mp4')

# %%

# ========
# Video IV
# ========

# choose what to plot
d = 200
alpha = 5
folder = 'tfi_200x1/tfi_1.0/'
iterations = 1001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 199.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 50, 100, 150, 200]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 50, 100, 150, 200])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$200 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$200 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_200x1_h=1.0_sr.mp4')

# %%

# =======
# Video V
# =======

# choose what to plot
d = 200
alpha = 5
folder = 'tfi_200x1/tfi_1.5/'
iterations = 1001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 199.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 50, 100, 150, 200]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 50, 100, 150, 200])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$200 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$200 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_200x1_h=1.5_rgn.mp4')

# %%

# ========
# Video VI
# ========

# choose what to plot
d = 200
alpha = 5
folder = 'tfi_200x1/tfi_1.5/'
iterations = 1001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 199.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 50, 100, 150, 200]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 50, 100, 150, 200])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$200 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$200 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_200x1_h=1.5_sr.mp4')

# %%

# =========
# Video VII
# =========

# choose what to plot
d = 20
alpha = 5
folder = 'tfi_20x20/tfi_2.0/'
iterations = 1001
normal = 0.1

# plot results
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d, d))

# initialization
t = 1000
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(3):
    for j in range(2):
        if i * 2 + j < alpha:
            colors = phase[i * 2 + j, :, :]
            alphas = size[i * 2 + j, :, :]
            im[i * 2 + j] = ax[i, j].imshow(colors, cmap, alpha=alphas,
                                            vmin=-np.pi, vmax=np.pi,
                                            aspect='auto')
            ax[i, j].set_xlim([-.5, 19.5])
            ax[i, j].get_xaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_xaxis().set_ticklabels([0, 10, 20])
            ax[i, j].set_ylim([-.5, 19.5])
            ax[i, j].get_yaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_yaxis().set_ticklabels([0, 10, 20])
            ax[i, j].tick_params(width=1.5, which='both')
fig.delaxes(ax[2, 1])
txt = fig.text(.28, 0.95, r'$20 \times 20$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :, :]
        alphas = size[i, :, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$20 \times 20$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_20x20_h=2.0_rgn.mp4')

# %%

# ==========
# Video VIII
# ==========

# choose what to plot
d = 20
alpha = 5
folder = 'tfi_20x20/tfi_2.0/'
iterations = 1001
normal = 0.1

# plot results
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d, d))

# initialization
t = 1000
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(3):
    for j in range(2):
        if i * 2 + j < alpha:
            colors = phase[i * 2 + j, :, :]
            alphas = size[i * 2 + j, :, :]
            im[i * 2 + j] = ax[i, j].imshow(colors, cmap, alpha=alphas,
                                            vmin=-np.pi, vmax=np.pi,
                                            aspect='auto')
            ax[i, j].set_xlim([-.5, 19.5])
            ax[i, j].get_xaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_xaxis().set_ticklabels([0, 10, 20])
            ax[i, j].set_ylim([-.5, 19.5])
            ax[i, j].get_yaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_yaxis().set_ticklabels([0, 10, 20])
            ax[i, j].tick_params(width=1.5, which='both')
fig.delaxes(ax[2, 1])
txt = fig.text(.28, 0.95, r'$20 \times 20$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :, :]
        alphas = size[i, :, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$20 \times 20$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_20x20_h=2.0_sr.mp4')

# %%

# ========
# Video IX
# ========

# choose what to plot
d = 20
alpha = 5
folder = 'tfi_20x20/tfi_3.0/'
iterations = 1001
normal = 0.1

# plot results
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d, d))

# initialization
t = 1000
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(3):
    for j in range(2):
        if i * 2 + j < alpha:
            colors = phase[i * 2 + j, :, :]
            alphas = size[i * 2 + j, :, :]
            im[i * 2 + j] = ax[i, j].imshow(colors, cmap, alpha=alphas,
                                            vmin=-np.pi, vmax=np.pi,
                                            aspect='auto')
            ax[i, j].set_xlim([-.5, 19.5])
            ax[i, j].get_xaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_xaxis().set_ticklabels([0, 10, 20])
            ax[i, j].set_ylim([-.5, 19.5])
            ax[i, j].get_yaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_yaxis().set_ticklabels([0, 10, 20])
            ax[i, j].tick_params(width=1.5, which='both')
fig.delaxes(ax[2, 1])
txt = fig.text(.28, 0.95, r'$20 \times 20$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :, :]
        alphas = size[i, :, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$20 \times 20$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_20x20_h=3.0_rgn.mp4')

# %%

# =======
# Video X
# =======

# choose what to plot
d = 20
alpha = 5
folder = 'tfi_20x20/tfi_3.0/'
iterations = 1001
normal = 0.1

# plot results
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d, d))

# initialization
t = 1000
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(3):
    for j in range(2):
        if i * 2 + j < alpha:
            colors = phase[i * 2 + j, :, :]
            alphas = size[i * 2 + j, :, :]
            im[i * 2 + j] = ax[i, j].imshow(colors, cmap, alpha=alphas,
                                            vmin=-np.pi, vmax=np.pi,
                                            aspect='auto')
            ax[i, j].set_xlim([-.5, 19.5])
            ax[i, j].get_xaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_xaxis().set_ticklabels([0, 10, 20])
            ax[i, j].set_ylim([-.5, 19.5])
            ax[i, j].get_yaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_yaxis().set_ticklabels([0, 10, 20])
            ax[i, j].tick_params(width=1.5, which='both')
fig.delaxes(ax[2, 1])
txt = fig.text(.28, 0.95, r'$20 \times 20$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :, :]
        alphas = size[i, :, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$20 \times 20$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_20x20_h=3.0_sr.mp4')

# %%

# ========
# Video XI
# ========

# choose what to plot
d = 20
alpha = 5
folder = 'tfi_20x20/tfi_4.0/'
iterations = 1001
normal = 0.1

# plot results
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d, d))

# initialization
t = 1000
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(3):
    for j in range(2):
        if i * 2 + j < alpha:
            colors = phase[i * 2 + j, :, :]
            alphas = size[i * 2 + j, :, :]
            im[i * 2 + j] = ax[i, j].imshow(colors, cmap, alpha=alphas,
                                            vmin=-np.pi, vmax=np.pi,
                                            aspect='auto')
            ax[i, j].set_xlim([-.5, 19.5])
            ax[i, j].get_xaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_xaxis().set_ticklabels([0, 10, 20])
            ax[i, j].set_ylim([-.5, 19.5])
            ax[i, j].get_yaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_yaxis().set_ticklabels([0, 10, 20])
            ax[i, j].tick_params(width=1.5, which='both')
fig.delaxes(ax[2, 1])
txt = fig.text(.28, 0.95, r'$20 \times 20$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :, :]
        alphas = size[i, :, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$20 \times 20$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_20x20_h=4.0_rgn.mp4')

# %%

# =========
# Video XII
# =========

# choose what to plot
d = 20
alpha = 5
folder = 'tfi_20x20/tfi_4.0/'
iterations = 1001
normal = 0.1

# plot results
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d, d))

# initialization
t = 1000
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(3):
    for j in range(2):
        if i * 2 + j < alpha:
            colors = phase[i * 2 + j, :, :]
            alphas = size[i * 2 + j, :, :]
            im[i * 2 + j] = ax[i, j].imshow(colors, cmap, alpha=alphas,
                                            vmin=-np.pi, vmax=np.pi,
                                            aspect='auto')
            ax[i, j].set_xlim([-.5, 19.5])
            ax[i, j].get_xaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_xaxis().set_ticklabels([0, 10, 20])
            ax[i, j].set_ylim([-.5, 19.5])
            ax[i, j].get_yaxis().set_ticks(np.array([0, 10, 20]) - .5)
            ax[i, j].get_yaxis().set_ticklabels([0, 10, 20])
            ax[i, j].tick_params(width=1.5, which='both')
fig.delaxes(ax[2, 1])
txt = fig.text(.28, 0.95, r'$20 \times 20$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :, :]
        alphas = size[i, :, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$20 \times 20$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/tfi_20x20_h=4.0_sr.mp4')

# %%

# ==========
# Video XIII
# ==========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_0.5/'
iterations = 5001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_h=0.5_rgn.mp4')

# %%

# =========
# Plot XIII
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_0.5/'
iterations = 5001
ticks = [0, 1000, 2000, 3000, 4000, 5000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_h=0.5_rgn.png', bbox_inches = 'tight', dpi = 200)

# %%

# =========
# Video XIV
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_0.5/'
iterations = 5001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_h=0.5_sr.mp4')

# %%

# =======
# Plot XIV
# =======

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_0.5/'
iterations = 5001
ticks = [0, 1000, 2000, 3000, 4000, 5000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_h=0.5_sr.png', bbox_inches = 'tight', dpi = 200)

# %%

# =========
# Video XV
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_0.5/'
iterations = 2001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_temper_h=0.5_rgn.mp4')

# %%

# ========
# Plot XV
# ========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_0.5/'
iterations = 2001
ticks = [0, 500, 1000, 1500, 2000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_temper_h=0.5_rgn.png', bbox_inches = 'tight', dpi = 200)

# %%

# =========
# Video XVI
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_0.5/'
iterations = 2001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')


# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_temper_h=0.5_sr.mp4')

# %%

# ========
# Plot XVI
# ========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_0.5/'
iterations = 2001
ticks = [0, 500, 1000, 1500, 2000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_temper_h=0.5_sr.png', bbox_inches = 'tight', dpi = 200)

# %%

# ==========
# Video XVII
# ==========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_1.0/'
iterations = 5001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_h=1.0_rgn.mp4')

# %%

# =========
# Plot XVII
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_1.0/'
iterations = 5001
ticks = [0, 1000, 2000, 3000, 4000, 5000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_h=1.0_rgn.png', bbox_inches = 'tight', dpi = 200)

# %%

# ===========
# Video XVIII
# ===========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_1.0/'
iterations = 5001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_h=1.0_sr.mp4')

# %%

# ==========
# Plot XVIII
# ==========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_1.0/'
iterations = 5001
ticks = [0, 1000, 2000, 3000, 4000, 5000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_h=1.0_sr.png', bbox_inches = 'tight', dpi = 200)

# %%

# =========
# Video XIX
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_1.0/'
iterations = 2001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_temper_h=1.0_rgn.mp4')

# %%

# ========
# Plot XIX
# ========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_1.0/'
iterations = 2001
ticks = [0, 500, 1000, 1500, 2000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_temper_h=1.0_rgn.png', bbox_inches = 'tight', dpi = 200)

# %%

# ========
# Video XX
# ========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_1.0/'
iterations = 2001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')


# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_temper_h=1.0_sr.mp4')

# %%

# =======
# Plot XX
# =======

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_1.0/'
iterations = 2001
ticks = [0, 500, 1000, 1500, 2000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_temper_h=1.0_sr.png', bbox_inches = 'tight', dpi = 200)

# %%

# =========
# Video XXI
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_1.5/'
iterations = 5001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_h=1.5_rgn.mp4')

# %%

# ========
# Plot XXI
# ========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_1.5/'
iterations = 5001
ticks = [0, 1000, 2000, 3000, 4000, 5000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_h=1.5_rgn.png', bbox_inches = 'tight', dpi = 200)

# %%

# ==========
# Video XXII
# ==========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_1.5/'
iterations = 5001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_h=1.5_sr.mp4')

# %%

# =========
# Plot XXII
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1/xxz_1.5/'
iterations = 5001
ticks = [0, 1000, 2000, 3000, 4000, 5000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_h=1.5_sr.png', bbox_inches = 'tight', dpi = 200)

# %%

# ===========
# Video XXIII
# ===========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_1.5/'
iterations = 2001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_temper_h=1.5_rgn.mp4')

# %%

# ==========
# Plot XXIII
# ==========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_1.5/'
iterations = 2001
ticks = [0, 500, 1000, 1500, 2000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'rgn_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_temper_h=1.5_rgn.png', bbox_inches = 'tight', dpi = 200)

# %%

# ==========
# Video XXIV
# ==========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_1.5/'
iterations = 2001
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
features = np.reshape(weights[:, :-alpha], (-1, alpha, d))

# initialization
t = 0
data = features[t, ...]
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[i, :][np.newaxis, :]
    alphas = size[i, :][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 99.5])
    ax[i].get_xaxis().set_ticks(np.array([0, 25, 50, 75, 100]) - .5)
    ax[i].get_xaxis().set_ticklabels([0, 25, 50, 75, 100])
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.28, 0.95, r'$100 \times 1$ chain, $t = ' + str(t) + '$',
                font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')


# update code
def updatefig(t):
    data = features[t, ...]
    size = np.abs(data) / normal
    size[size > 1] = 1
    phase = np.angle(data)
    for i in range(alpha):
        colors = phase[i, :][np.newaxis, :]
        alphas = size[i, :][np.newaxis, :]
        im[i].set_data(colors)
        im[i].set_alpha(alphas)
    txt.set_text(r'$100 \times 1$ chain, $t = ' + str(t) + '$')
    return txt, im

anim = animation.FuncAnimation(fig, updatefig, frames=iterations, interval=15)
anim.save('videos/xxz_100x1_temper_h=1.5_sr.mp4')

# %%

# =========
# Plot XXIV
# =========

# choose what to plot
d = 100
alpha = 5
folder = 'xxz_100x1_temper/xxz_1.5/'
iterations = 2001
ticks = [0, 500, 1000, 1500, 2000]
normal = 0.5

# plot results
fig, ax = plt.subplots(5, 1, figsize=(10, 8))
weights = np.load(folder + 'sr_weights.npy')
data = np.reshape(weights[:, :alpha], (-1, alpha))
size = np.abs(data) / normal
size[size > 1] = 1
phase = np.angle(data)
im = [None] * alpha
for i in range(alpha):
    colors = phase[:, i][np.newaxis, :]
    alphas = size[:, i][np.newaxis, :]
    im[i] = ax[i].imshow(colors, cmap, alpha=alphas, vmin=-np.pi,
                          vmax=np.pi, aspect='auto')
    ax[i].set_xlim([-.5, 1000.5])
    ax[i].get_xaxis().set_ticks(np.array(ticks) - .5)
    ax[i].get_xaxis().set_ticklabels(ticks)
    ax[i].set_ylim([-.5, .5])
    ax[i].get_yaxis().set_ticks([])
    ax[i].tick_params(width=1.5, which='both')
txt = fig.text(.33, 0.95, r'$100 \times 1$ chain', font, ha='left')
fig.tight_layout()
fig.subplots_adjust(top=0.9)

# color bar
cbar = fig.colorbar(im[0], ax=ax, location='right', shrink=.9,
                    ticks=[-np.pi, 0, np.pi])
cbar.ax.set_yticklabels([r'$-\pi$', r'0', r'$\pi$'])
cbar.ax.set_ylabel('Phase', rotation=90)
cbar.ax.tick_params(width=1.5, which='both')

# save
fig.savefig('videos/xxz_100x1_temper_h=1.5_sr.png', bbox_inches = 'tight', dpi = 200)
