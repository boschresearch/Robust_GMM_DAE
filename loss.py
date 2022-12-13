"total loss  for training the model"
"code adapted but modified from https://github.com/boschresearch/GMM_DAE"
# ----------------------------------------------------------------------------
import numpy as np
from torch.nn import functional as F

from robust_latent_regularizer import (
    mean_squared_kolmogorov_smirnov_distance_extendedgmm,
    mean_squared_twopointks,
    mean_squared_covariance_extendedgmm
)
from utils import draw_gmm_samples


def estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, num_samples=100):
    """Estimated the weights of our robust multi-modal loss."""
    _, dimension = gmm_centers.shape
    ks_losses, cv_losses, ks_pairlosses = [], [], []

    # Estimate wieghts with gmm samples:
    for i in range(num_samples):
        z, _ = draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        z1, _ = draw_gmm_samples(
            batch_size, gmm_centers, gmm_std
        )
        ks_loss, xdist_loss = mean_squared_kolmogorov_smirnov_distance_extendedgmm(z, z1, gmm_centers=gmm_centers,
                                                                                   gmm_std=gmm_std)
        ks_loss = ks_loss.cpu().detach().numpy()
        ks_pairloss = mean_squared_twopointks(z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std)
        ks_pairloss = ks_pairloss.cpu().detach().numpy()
        cv_loss = mean_squared_covariance_extendedgmm(
            z, z1, gmm_centers=gmm_centers, gmm_std=gmm_std
        )
        cv_loss = cv_loss.cpu().detach().numpy()

        ks_losses.append(ks_loss)
        ks_pairlosses.append(ks_pairloss)
        cv_losses.append(cv_loss)

    ks_weight = 1 / np.mean(ks_losses)
    ks_pair_weight = 1 / np.mean(ks_pairlosses)
    cv_weight = 1 / np.mean(cv_losses)
    return ks_weight, ks_pair_weight, cv_weight


def get_vaeloss(predicted_images, latent_vectors, true_images, ks_weight, cv_weight, image_loss_weight,
                gmm_centers,
                gmm_std, is_invcdf, adv_latent_vectors, ks_pair_weight):
    """Estimated the total training object of our robust multi-modal deterministic autoencoder."""
    image_loss = F.mse_loss(predicted_images, true_images)

    ks_loss, xloss = mean_squared_kolmogorov_smirnov_distance_extendedgmm(latent_vectors, adv_latent_vectors,
                                                                          gmm_centers, gmm_std)
    ks_pairloss = mean_squared_twopointks(latent_vectors, adv_latent_vectors, gmm_centers, gmm_std)

    extended_covloss = mean_squared_covariance_extendedgmm(latent_vectors, adv_latent_vectors, gmm_centers, gmm_std)
    weighted_ksloss = ks_weight * ks_loss
    weighted_kspair_loss = ks_pair_weight * ks_pairloss
    weighted_cov_loss = cv_weight * extended_covloss

    weighted_imageloss = image_loss_weight * image_loss

    losses = weighted_ksloss + weighted_cov_loss + weighted_imageloss + weighted_kspair_loss
    loss_mean = losses.mean().cuda()
    return loss_mean, weighted_ksloss, weighted_cov_loss, weighted_imageloss, weighted_kspair_loss
