"code adapted but modified from https://github.com/FabianBarrett/Lipschitz_VAEs/blob/master/lnets/tasks/vae/mains/latent_space_attack.py "
# Implements latent space attacks for quantitaive evaluation of VAE robustness
import matplotlib

matplotlib.use('Agg')
import scipy.optimize
from prepareinput import prepare_data_loader
import argparse
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import os
from model import VAE
import torch
import torch.nn as nn
from attackutils import get_target_image, sample_d_ball, get_target_image_celeb
import pytorch_msssim

loss = torch.nn.MSELoss()
attack_losses = []
recon_dist = []
norm_recon_dist = []
mse = []
msssim = []
inp_dist = []
inp_msssim = []
rmsssim = []
amsssim = []
recon_dist_targetrecon = []


def latent_space_optimize_noise(model, img_size, image, target_image, initial_noise, nc, soft=False,
                                regularization_coefficient=None, maximum_noise_norm=None):
    adversarial_losses = []

    def fmin_func(noise):

        # Soft determines whether latent space attack objective should be regularization_coefficient * norm of noise
        if soft:
            loss, gradient = model.module.eval_latent_space_attack(image[None, :], target_image[None, :],
                                                                   noise[None, :], soft=soft,
                                                                   regularization_coefficient=regularization_coefficient)
        # If not, use hard constraint on norm of noise (i.e. attack is limited to this norm)
        else:
            loss, gradient = model.module.eval_latent_space_attack(image[None, :], target_image[None, :],
                                                                   noise[None, :], soft=soft,
                                                                   maximum_noise_norm=maximum_noise_norm)
        adversarial_losses.append(loss)
        return float(loss.data.cpu().detach().numpy()), gradient.data.cpu().detach().numpy().flatten().astype(
            np.float64)

    # Bounds on the noise to ensure pixel values remain in interval [0, 1]
    lower_limit = -image.data.numpy().flatten()
    upper_limit = (1.0 - image.data.numpy().flatten())

    bounds = zip(lower_limit, upper_limit)
    bounds = [sorted(y) for y in bounds]

    # BB: Optimizer to find adversarial noise
    noise, attack_loss, _ = scipy.optimize.fmin_l_bfgs_b(fmin_func,
                                                         x0=initial_noise,
                                                         bounds=bounds,
                                                         m=25,
                                                         factr=10)
    attack_losses.append(attack_loss)

    return (torch.tensor(noise).view(1, nc, img_size, img_size)).float(), adversarial_losses, attack_losses


def get_attack_images(model, img_size, original_image, target_image, initial_noise, nc, soft=False,
                      regularization_coefficient=None, maximum_noise_norm=None, modeltype=None):
    if modeltype == "VAE":
        clean_reconstruction, _, _ = model(original_image[None, :].cuda())
    else:
        clean_reconstruction, _, _, _ = model(original_image[None, :].cuda())
    reshaped_clean_reconstruction = clean_reconstruction.view(1, nc, img_size, img_size)
    noise, _, attack_losses = latent_space_optimize_noise(model, img_size, original_image, target_image.cuda(),
                                                          initial_noise, nc, soft=soft,
                                                          regularization_coefficient=regularization_coefficient,
                                                          maximum_noise_norm=maximum_noise_norm)
    if not soft:
        noise = (maximum_noise_norm * noise.div(noise.norm(p=2))) if (noise.norm(p=2) > maximum_noise_norm) else noise

    noisy_image = original_image + noise.view(1, nc, img_size, img_size)
    if modeltype == "VAE":
        noisy_reconstruction, _, _ = model(noisy_image.cuda())
    else:
        noisy_reconstruction, _, _, _ = model(noisy_image.cuda())
    reshaped_noisy_reconstruction = noisy_reconstruction.view(1, nc, img_size, img_size)
    target_image = target_image.cuda()
    target_recon, _, _, _ = model(target_image.view(1, nc, img_size, img_size).cuda())
    reshaped_target_reconstruction = target_recon.view(1, nc, img_size, img_size)
    if nc == 1:
        image_compilation = torch.cat((original_image.view(1, nc, img_size, img_size).cuda(),
                                       reshaped_clean_reconstruction.cuda(),
                                       noisy_image.cuda(),
                                       reshaped_noisy_reconstruction.cuda(),
                                       target_image.view(1, nc, img_size, img_size)), dim=-1)
    else:
        image_compilation = torch.cat((original_image.permute(1, 2, 0).cuda(),
                                       clean_reconstruction.squeeze().permute(1, 2, 0).cuda(),
                                       noisy_image.squeeze().permute(1, 2, 0).cuda(),
                                       noisy_reconstruction.squeeze().permute(1, 2, 0).cuda(),
                                       target_image.permute(1, 2, 0)), dim=-2)
    recon_dist.append(
        torch.norm(torch.flatten(target_image.unsqueeze(0)) - torch.flatten(reshaped_noisy_reconstruction),
                   p=2).detach().cpu().numpy())
    den = torch.norm(torch.flatten(target_image.unsqueeze(0)) - torch.flatten(reshaped_clean_reconstruction),
                     p=2)
    num = torch.norm(torch.flatten(target_image.unsqueeze(0)) - torch.flatten(reshaped_noisy_reconstruction),
                     p=2)
    norm_recon_dist.append((num / den).detach().cpu().numpy())
    mse.append(loss(reshaped_noisy_reconstruction, target_image.unsqueeze(0)).detach().cpu().numpy())
    msssim.append(pytorch_msssim.msssim(reshaped_noisy_reconstruction, target_image.unsqueeze(0), window_size=6,
                                        normalize='relu').detach().cpu().numpy())
    inp_dist.append(torch.norm(torch.flatten(original_image.unsqueeze(0)) - torch.flatten(noisy_image),
                               p=2).detach().cpu().numpy())
    inp_msssim.append(pytorch_msssim.msssim(noisy_image, original_image.unsqueeze(0), window_size=6,
                                            normalize='relu').detach().cpu().numpy())
    recon_dist_targetrecon.append(
        torch.norm(torch.flatten(reshaped_noisy_reconstruction) - torch.flatten(reshaped_target_reconstruction),
                   p=2).detach().cpu().numpy())
    noisy_image = noisy_image.cuda()
    target_image = target_image.cuda()
    amsssim.append(pytorch_msssim.msssim(reshaped_noisy_reconstruction, noisy_image, window_size=6,
                                         normalize='relu').detach().cpu().numpy())
    rmsssim.append(pytorch_msssim.msssim(reshaped_target_reconstruction, target_image.unsqueeze(0), window_size=6,
                                         normalize='relu').detach().cpu().numpy())
    return image_compilation, attack_losses, recon_dist, norm_recon_dist, mse, msssim, inp_dist, inp_msssim, recon_dist_targetrecon, rmsssim, amsssim


def latent_space_attack(model, iterator, img_size, num_images, dataset, nc, d_ball_init=True, soft=False,
                        regularization_coefficient=None, maximum_noise_norm=None, plotting_dir=None, modeltype=None):
    sample = next(iter(iterator))
    attack_sample = (sample[0][:num_images], sample[1][:num_images])

    for i in range(num_images):
        # Get original and target images
        index = 25
        original_image, original_class = attack_sample[0][index], attack_sample[1][index]
        if dataset == "CELEB":
            target_image, target_class = get_target_image_celeb(attack_sample, index, num_images)
        else:
            target_image, target_class = get_target_image(attack_sample, original_class, index, num_images)

        if d_ball_init:
            # Sample initial noise for adversarial attack (same noise for both Lipschitz and comparison model)
            initial_noise = sample_d_ball(img_size * img_size * nc, maximum_noise_norm).reshape(
                (1, nc, img_size, img_size)).astype(np.float32)

        else:
            # Sample initial noise for adversarial attack (same noise for both Lipschitz and comparison model)
            initial_noise = np.random.uniform(-1e-8, 1e-8,
                                              size=(1, nc, img_size, img_size)).astype(np.float32)

        # Perform adversarial attack and get related images

        image_compilation, attack_losses, recon_dist, normrecon_dist, MSE, MSSSIM, inpdist, inpmsssim, recondisttargetrecon, RMSSSIM, AMSSSIM = get_attack_images(
            model, img_size, original_image, target_image,
            initial_noise, nc, soft=soft,
            regularization_coefficient=regularization_coefficient,
            maximum_noise_norm=maximum_noise_norm, modeltype=modeltype)

        # Plotting
        plt.figure(figsize=(9, 3))
        if dataset == "MNIST" or dataset == "FASHIONMNIST":
            plt.imshow(image_compilation.detach().cpu().squeeze().numpy(), vmin=0, vmax=1, cmap="gray")
        else:
            plt.imshow((image_compilation.detach().cpu().numpy() * 255).astype(np.uint8))
        plt.axis('off')
        savedir = plotting_dir + "/latent_space_hardattacks/"
        os.makedirs(savedir, exist_ok=True)
        plt.savefig(
            savedir + "/latent_attack_{}_maximum_perturbation_norm_{}.png".format(i + 1,
                                                                                  maximum_noise_norm), dpi=300,
            bbox_inches='tight')
    result_txt = plotting_dir + "/metrics.txt"
    with open(result_txt, "a+") as myfile:
        myfile.write(str("maximum_noise_norm - ") + str(maximum_noise_norm) + "\n")
        myfile.write(str("attack_losses - ") + str(np.mean(attack_losses)) + "\n")
        myfile.write(str("recon_dist - ") + str(np.mean(recon_dist)) + "\n")
        myfile.write(str("MSE - ") + str(np.mean(MSE)) + "\n")
        myfile.write(str("MSSSIM - ") + str(np.mean(MSSSIM)) + "\n")
        myfile.write(str("inpmsssim - ") + str(np.mean(inpmsssim)) + "\n")
        myfile.write(str("recondisttargetrecon - ") + str(np.mean(recondisttargetrecon)) + "\n")


def latent_attack_model(opt):
    model_exp_dir = opt['modelpath']
    modelpath = model_exp_dir
    latentsize = opt['latentsize']
    image_channel = opt['image_channel']
    img_size = opt['img_size']
    modeltype = opt['modeltype']
    dataset = opt['dataset']
    datadir = opt['datadir']
    model = VAE(dataset=dataset, nc=image_channel, ndf=128, nef=128, nz=latentsize, isize=img_size,
                is_train=True, latent_noise_scale=0.4)
    print("model loaded!")
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(modelpath))

    if opt['data']['cuda']:
        print('Using CUDA')
        model.cuda()

    trainloader, testloader, valloader = prepare_data_loader(datadir, 128, 128, 128, dataset=dataset)
    model.eval()

    latent_space_attack(model, testloader, img_size,
                        opt['num_images'], dataset, image_channel, d_ball_init=opt['d_ball_init'], soft=opt['soft'],
                        regularization_coefficient=opt['regularization_coefficient'],
                        maximum_noise_norm=opt['maximum_noise_norm'], plotting_dir=model_exp_dir, modeltype=modeltype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack trained VAE')
    parser.add_argument('--modelpath', type=str, metavar='model path',
                        help="location of pretrained model weights to evaluate")
    parser.add_argument('--modeltype', type=str, metavar='model type',
                        help="type of pretrained model to evaluate")
    parser.add_argument('--datadir', type=str, metavar='data directory',
                        help="type of pretrained model to evaluate")
    parser.add_argument('--dataset', type=str,
                        help="dataset name to use")
    parser.add_argument('--savedir', type=str,
                        help="path to save results")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--num_images', type=int, default=10, help='number of images to perform latent space attack on')
    parser.add_argument('--latentsize', type=int, default=10,
                        help='size of the ls of the model to perform latent space attack on')
    parser.add_argument('--img_size', type=int, default=32, help='size of images to perform latent space attack on')
    parser.add_argument('--image_channel', type=int, default=1,
                        help='number of channels in images to perform latent space attack on')
    parser.add_argument('--soft', type=bool, default=False,
                        help='whether latent attack should feature soft constraint on noise norm (hard constraint if '
                             'False)')
    parser.add_argument('--d_ball_init', type=bool, default=True,
                        help='whether attack noise should be initialized from random point in d-ball around image ('
                             'True/False)')
    parser.add_argument('--regularization_coefficient', type=float, default=1.0,
                        help='regularization coefficient to use in latent space attack')
    parser.add_argument('--maximum_noise_norm', type=float, default=3,
                        help='maximal norm of noise in max damage attack')

    args = vars(parser.parse_args())

    opt = {}
    for k, v in args.items():
        cur = opt
        tokens = k.split('.')
        for token in tokens[:-1]:
            if token not in cur:
                cur[token] = {}
            cur = cur[token]
        cur[tokens[-1]] = v

    latent_attack_model(opt)
