"Train the model"
"code adapted but modified from https://github.com/boschresearch/GMM_DAE"
# ----------------------------------------------------------------------------
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

from loss import (get_vaeloss, estimate_loss_coefficients)
from model import VAE
from dataloader import prepare_data_loadernpz
from utils import set_gmm_centers, exp_lr_scheduler

if __name__ == "__main__":
    try:
        from configparser import ConfigParser
    except ImportError:
        from configparser import ConfigParser  # ver. < 3.0
    major_idx = str(sys.argv[1])
    # instantiate
    config = ConfigParser()

    # parse existing file
    config.read('config.ini')
    dataset = config.get(major_idx, 'dataset')
    experiment_name = config.get(major_idx, 'experiment_name')
    img_size = config.getint(major_idx, 'image_size')
    batch_size = config.getint(major_idx, 'batch_size')
    num_cluster = config.getint(major_idx, 'num_clusters')
    epochs = config.getint(major_idx, 'epochs')
    latent_dim = config.getint(major_idx, 'latent_dim')
    image_num_channels = config.getint(major_idx, 'image_num_channels')
    nef = config.getint(major_idx, 'nef')
    ndf = config.getint(major_idx, 'ndf')
    lr = config.getfloat(major_idx, 'lr')
    coup = config.getfloat(major_idx, 'coup')
    epsilon = config.getfloat(major_idx, 'epsilon')
    alpha = config.getfloat(major_idx, 'alpha')
    exp_lr = config.getboolean(major_idx, 'exp_lr')
    is_noise = config.getboolean(major_idx, 'is_noise')
    latent_noise_scale = config.getfloat(major_idx, 'latent_noise_scale')
    save_dir = config.get(major_idx, 'save_dir') + "/" + dataset + "/" + experiment_name
    data_dir = config.get(major_idx, 'data_dir')
    image_loss_weight = config.getfloat(major_idx, 'image_loss_weight')
    resume = config.getboolean(major_idx, 'resume')
    resume_path = config.get(major_idx, 'resume_path')

    # get train and test dataloder
    trainloader, testloader, valloader, classes = prepare_data_loadernpz(data_dir, dataset, image_num_channels,
                                                                         batch_size, batch_size)
    # set prior means and std
    gmm_centers, gmm_std = set_gmm_centers(latent_dim, num_cluster)

    # get weights of the loss functions used
    ks_weight, ks_pair_weight, cv_weight = estimate_loss_coefficients(batch_size, gmm_centers, gmm_std, coup, num_samples=100)

    # Initialize the model.
    model = VAE(nc=image_num_channels, ndf=ndf, nef=nef, nz=latent_dim, isize=img_size,
                latent_noise_scale=latent_noise_scale, is_train=is_noise, gmm_centers=gmm_centers, gmm_std=gmm_std)
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if resume:
        print("resuming training")
        model.load_state_dict(torch.load(resume_path))
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)
    for epoch in range(epochs):  # loop over the dataset multiple times
        if exp_lr:
            optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=7)
        for i, (data, target) in enumerate(trainloader, 0):
            model.train()
            inputs = Variable(data.type(torch.cuda.FloatTensor))
            # adversarial training FGSM to generate advesarial samples
            adversarial_samples = model.module.craft_PGDadversarial_samples(inputs, epsilon, alpha)
            recon_images, adv_recon_images, latent_vectors, adv_latent_vectors = model(inputs, adversarial_samples)
            loss_mean, weighted_ksloss, weighted_xloss, weighted_cov_loss, weighted_imageloss, weighted_kspairloss = \
                get_vaeloss(recon_images, latent_vectors, inputs, ks_weight, cv_weight, image_loss_weight,
                            gmm_centers, gmm_std, adv_latent_vectors, ks_pair_weight, coup)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

    print('Finished Training')
    # Save the model
    torch.save(model.state_dict(), '%s/vae_epoch_final_%d.pth' % (save_dir, epoch))
    print('The trained model was stored in %s' % save_dir)
