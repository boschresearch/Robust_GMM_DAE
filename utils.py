"helper scripts to sample from the GMM prior"
"code from https://github.com/boschresearch/GMM_DAE"
# ----------------------------------------------------------------------------
import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sobol_seq import i4_sobol_generate_std_normal


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def draw_gmm_samples(num_samples, gmm_centers, gmm_std):
    """Draws a given number of samples from a GMM with given centers and per center standard deviation under
    the assumption that different dimensions are uncorrelated on a per center level and equal weighing of modes.

    Will be used for weight estimation"""
    num_gmm_centers, dimension = gmm_centers.shape

    samples = []
    components = []
    for _ in range(num_samples):
        component = np.random.choice(range(num_gmm_centers))

        component_mean = gmm_centers[component, :]
        component_cov = torch.eye(dimension) * gmm_std

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=component_mean.cuda(), covariance_matrix=component_cov.cuda()
        )

        sample = distribution.sample((1,))
        samples.append(sample)
        components.append(component)
    samples = torch.vstack(samples)

    return samples, components


def sample_gmm_percluster(cluster_index, gmm_centers, gmm_std, nb_samples=500):
    samples = []
    num_gmm_centers, dimension = gmm_centers.shape
    for _ in range(nb_samples):
        component_mean = gmm_centers[cluster_index, :]
        component_cov = torch.eye(dimension) * gmm_std

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=component_mean.cuda(), covariance_matrix=component_cov.cuda()
        )

        sample = distribution.sample((1,))
        samples.append(sample)

    samples = torch.vstack(samples)

    return samples


def set_gmm_centers(dimension, num_gmm_components):
    """
    Defines our prior per dimension for all plots.
    """
    gmm_centers = []
    mu = np.zeros(dimension)
    mu[0] = 10
    for i in range(0, num_gmm_components):
        gmm_centers.append(np.roll(mu, i))
    gmm_std = 1.0
    gmm_centers = torch.tensor(gmm_centers).cuda().float()
    return gmm_centers, gmm_std


def set_gmm_centers_sobol(dimension, num_gmm_components, spread_modes=True):
    """
    Defines our prior per dimension for all plots.
    """

    gmm_centers = i4_sobol_generate_std_normal(
        dimension, num_gmm_components, skip=1
    ).astype("float32")
    gmm_centers = torch.as_tensor(gmm_centers)
    gmm_centers /= torch.norm(gmm_centers, p=2, dim=1, keepdim=True)
    gmm_std = 1
    min_distance = float("inf")
    for c1, c2 in combinations(gmm_centers, 2):
        min_distance = min(torch.norm(c1 - c2, p=2), min_distance)

    if min_distance == 0:
        print('Could not generated %d centers in %d dimensions, min distance was zero...' % (
            num_gmm_components, dimension))
        return None, None

    if spread_modes:
        # Spread centers until a desired minimum distance is reached.
        while min_distance < 6:
            print('Minimum distance between prior means was %f, multiply means by a factor of two...' % (min_distance))
            gmm_centers *= 2
            min_distance = float("inf")

            for c1, c2 in combinations(gmm_centers, 2):
                min_distance = min(torch.norm(c1 - c2, p=2), min_distance)
    gmm_centers = torch.tensor(gmm_centers).cuda().float()
    return gmm_centers, gmm_std
