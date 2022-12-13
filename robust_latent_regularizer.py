"Proposed latent regularization for training the model"
"code adapted but modified from https://github.com/boschresearch/GMM_DAE"
# ----------------------------------------------------------------------------
import matplotlib
import torch
import numpy as np


def compute_gmm_covariance(gmm_centers, gmm_std):
    """Computes the full covariance matrix of a GMM model with given centers and per center standard deviation under
    the assumption that different dimensions are uncorrelated on a per center level and equal weighing of modes."""

    num_gmm_centers, dimension = gmm_centers.shape
    component_cov = torch.eye(dimension) * gmm_std

    # Weighted ceters = mean due to equal weighin
    weighted_gmm_centers = gmm_centers.mean(axis=0)
    gmm_centers = gmm_centers - weighted_gmm_centers

    # Implementing Law of total variance:;
    # Conditional Expectation:
    conditional_expectation = 0
    for component in range(num_gmm_centers):
        center_mean = gmm_centers[component, :].reshape(dimension, 1)
        conditional_expectation += (1 / num_gmm_centers) * torch.mm(center_mean, center_mean.t())
    # Expected conditional variance equals component_cov, since all components are weighted equally,
    # and all component covariances are the same.
    return component_cov.cuda(), component_cov.cuda() + conditional_expectation.cuda()


def compute_empirical_covariance(embedding_matrix):
    """Computes empirical covariance matrix of a given embedding matrix."""
    m = torch.mean(embedding_matrix, dim=0)
    sigma = (
            torch.mm((embedding_matrix - m).t(), (embedding_matrix - m))
            / embedding_matrix.shape[0]
    )
    return sigma


def mean_squared_kolmogorov_smirnov_distance_extendedgmm(embedding_matrix, adv_embedding_matrix, gmm_centers, gmm_std):
    """Return the kolmogorov distance for each dimension.

    embedding_matrix:
        The latent representation of the batch.
    adv embedding_matrix:
        The latent representation of the adversarial sample batch.
    gmm_centers:
        Centers of the GMM components in that space. All are assumed to have the same weight
    gmm_std:
        All components of the GMM are assumed to have share the same covariance matrix: C = gmm_std**2 * Identity.

    Note that the returned distances are NOT in the same order as embedding matrix.
    Thus, this is useful for means/max, but not for visual inspection.
    """
    total_embedding = torch.cat((embedding_matrix, adv_embedding_matrix), dim=0)
    sorted_embeddings = torch.sort(total_embedding, dim=-2).values
    emb_num, emb_dim = sorted_embeddings.shape[-2:]
    num_gmm_centers, _ = gmm_centers.shape
    # For the sorted embeddings, the empirical CDF depends to the "index" of each
    # embedding (the number of embeddings before it).
    # Unsqueeze enables broadcasting
    empirical_cdf = torch.linspace(
        start=1 / emb_num,
        end=1.0,
        steps=emb_num,
        device=embedding_matrix.device,
        dtype=embedding_matrix.dtype,
    ).unsqueeze(-1)

    # compute CDF values for the embeddings using the Error Function
    normalized_embedding_distances_to_centers = (sorted_embeddings[:, None] - gmm_centers[None]) / gmm_std
    normal_cdf_per_center = 0.5 * (1 + torch.erf(normalized_embedding_distances_to_centers * 0.70710678118))
    normal_cdf = normal_cdf_per_center.mean(dim=1)
    return torch.nn.functional.mse_loss(normal_cdf, empirical_cdf)


def mean_squared_twopointks(embedding_matrix, adv_embeddingmatrix, gmm_centers, gmm_std):
    """Return the 2-point kolmogorov distance loss for each dimension.

    embedding_matrix:
        The latent representation of the batch.
    gmm_centers:
        Centers of the GMM components in that space. All are assumed to have the same weight
    gmm_std:
        Standard deviation of the GMM components.
    """
    sorted_embeddings = torch.sort(embedding_matrix, dim=-2).values
    sorted_advembeddings = torch.sort(adv_embeddingmatrix, dim=-2).values
    num_gmm_centers, _ = gmm_centers.shape

    # compute CDF values for the embeddings using the Error Function
    normalized_embedding_distances_to_centers = (sorted_embeddings[:, None] - gmm_centers[None]) / gmm_std
    normal_cdf_per_center = 0.5 * (1 + torch.erf(normalized_embedding_distances_to_centers * 0.70710678118))
    normal_cdf = normal_cdf_per_center.mean(dim=1)

    normalized_advembedding_distances_to_centers = (sorted_advembeddings[:, None] - gmm_centers[None]) / gmm_std
    normal_advcdf_per_center = 0.5 * (1 + torch.erf(normalized_advembedding_distances_to_centers * 0.70710678118))
    normal_advcdf = normal_advcdf_per_center.mean(dim=1)
    return torch.nn.functional.mse_loss(normal_cdf, normal_advcdf)


def mean_squared_covariance_extendedgmm(embedding_matrix, adv_embedding_matrix, gmm_centers, gmm_std, coup):
    """Compute mean squared distance between the empirical covariance matrix of a (embedding matrix, adversarial
    embedding matrix) and the covariance of a GMM prior with given centers and per center standard deviation under
    the assumption that different dimensions are uncorrelated on a per center level and equal weighing of modes.

    Parameters
    ----------
    embedding_matrix: torch.Tensor
    adv_embedding_matrix: torch.Tensor
        Latent Vectors.
    gmm_centers:
        Centers of the GMM components in that space. All are assumed to have the same weight
    gmm_std:
        Standard deviation of the GMM components.
    Returns
    -------
    mean_cov: float
        Mean squared distance between empirical and prior covariance.

    """
    # Compute empirical covariances:
    comp_covariance, gmm_covariance = compute_gmm_covariance(gmm_centers, gmm_std)
    comp_covariance.to(embedding_matrix.device)
    gmm_covariance.to(embedding_matrix.device)

    # Compute cross covariances:
    cross_covar = torch.eye(gmm_covariance.size(dim=0)).cuda()
    cross_covar.to(embedding_matrix.device)
    gmm_covariance1 = torch.cat((gmm_covariance, coup * gmm_covariance), dim=0)
    gmm_covariance2 = torch.cat((coup * gmm_covariance, gmm_covariance), dim=0)
    combined_gmm_covariance = torch.cat((gmm_covariance1, gmm_covariance2), dim=1)
    combined_empirical = torch.cat((embedding_matrix, adv_embedding_matrix), dim=1)
    sigma_combined = compute_empirical_covariance(combined_empirical)
    diff = torch.pow(sigma_combined - combined_gmm_covariance, 2)
    mean_cov = torch.mean(diff)
    return mean_cov
