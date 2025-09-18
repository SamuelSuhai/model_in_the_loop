import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from alphaanalysis.plot import plot_scale_bar


def normalize_stack(stack, q_low=5, q_high=0.05):
    norm_stack0 = stack.astype(float).copy()
    p_low = np.nanpercentile(norm_stack0, q=q_low)
    norm_stack0 -= p_low
    norm_stack0[norm_stack0 < 0] = 0.

    p_high = np.nanpercentile(norm_stack0, q=100 - q_high)
    norm_stack0 /= p_high
    norm_stack0[norm_stack0 > 1] = 1.

    return norm_stack0


def get_col_stack(norm_stack, color='lime', gamma=1.):
    """Project stack to RGB color space."""
    cmap = LinearSegmentedColormap.from_list(color, ['k', color], N=256)
    col_stack = cmap(norm_stack ** gamma)[:, :, :, :3]
    return col_stack


def merge_col_stacks(col_stacks, norm_stacks):
    """Merge multiple colored stacks into one."""
    non_zero = np.sum(norm_stacks, axis=0) > 0
    weights_sum = np.sum([norm_stack[non_zero] for norm_stack in norm_stacks], axis=0)

    col_stack_merged = np.zeros(col_stacks[0].shape)

    for norm_stack, col_stack in zip(norm_stacks, col_stacks):
        weights = np.zeros(norm_stack.shape)
        weights[non_zero] = norm_stack[non_zero] / weights_sum
        weights = np.tile(weights[:, :, :, np.newaxis], (1, 1, 1, 3))
        col_stack_merged += weights * col_stack

    return col_stack_merged


def proj_z_col_stack(col_stack, gammaz=2., offz=1.):
    """Project colored stack to z-axis."""
    # Get weights in z-direction
    z_weights = np.linspace(0, 1, col_stack.shape[2]) + offz
    z_weights = z_weights ** gammaz
    z_weights /= np.mean(z_weights)

    col_stack_proj_z = np.clip(np.max(col_stack * np.tile(z_weights[:, np.newaxis], (1, 3)), axis=2), 0, 1)

    return col_stack_proj_z


def get_col_stack_merged_z(norm_stacks, colors, gammas=None, gammaz=2., offz=1., plot=False):
    """Merge multiple stacks into one colored channel and project to z-axis."""
    assert len(norm_stacks) == len(colors)

    if gammas is None:
        gammas = [1.] * len(norm_stacks)

    col_stacks = [get_col_stack(norm_stack, color=color, gamma=gamma) for norm_stack, color, gamma in
                  zip(norm_stacks, colors, gammas)]

    col_stack_merged = merge_col_stacks(col_stacks, norm_stacks)
    col_stack_z_proj = proj_z_col_stack(col_stack_merged, gammaz=gammaz, offz=offz)

    if plot:
        plot_col_stack_z_proj(col_stacks, col_stack_z_proj)

    return col_stack_z_proj, col_stacks


def plot_col_stack_z_proj(col_stacks, col_stack_z_proj):
    fig, axs = plt.subplots(1, len(col_stacks) + 1, figsize=(12, 3), sharex='all', sharey='all')
    for ax, col_stack in zip(axs[:-1], col_stacks):
        ax.imshow(np.max(col_stack, axis=2))
    axs[-1].imshow(col_stack_z_proj)
    return fig, axs


def get_col_z_proj(norm_stack0, norm_stack1, gamma0=1., gamma1=0.8, gammaz=2., offz=1.,
                   colors=('lime', 'magenta'), plot=False, return_col_stacks=False):
    col_stack_merged_z, col_stacks = get_col_stack_merged_z(
        norm_stacks=[norm_stack0, norm_stack1], colors=colors, gammas=[gamma0, gamma1],
        gammaz=gammaz, offz=offz, plot=plot)

    if return_col_stacks:
        return col_stack_merged_z, col_stacks
    else:
        return col_stack_merged_z


def plot_col_z_proj(ax, col_z_proj, n_artifact=30, pixel_size_um=1., center=True, rotate=True):
    w, h = col_z_proj[n_artifact:, :].shape[:2]

    if center:
        extent = np.array([-w / 2, w / 2, -h / 2, h / 2]) * pixel_size_um
    else:
        extent = np.array([0, w, 0, h]) * pixel_size_um

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')

    ax.set_facecolor('k')

    if rotate:
        im = np.swapaxes(col_z_proj[n_artifact:, :], 0, 1)
    else:
        im = col_z_proj[n_artifact:, :]

    ax.imshow(im, origin='lower', extent=extent)


def plot_3d_scatter(norm_stack1, norm_stack0):
    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    x, y, z = np.where(norm_stack1 > 0.25)
    ax.scatter(x, y, z, s=0.4, c='magenta')

    x, y, z = np.where(norm_stack0 > 0.25)
    ax.scatter(x, y, z, s=0.4, c='green')

    ax.view_init(elev=20., azim=-35, roll=0)
    return ax
