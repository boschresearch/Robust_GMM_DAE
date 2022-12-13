"code adapted but modified from https://github.com/FabianBarrett/Lipschitz_VAEs/blob/master/lnets/tasks/vae/mains/latent_space_attack.py"
# Implements maximum damage attack for quantitative evaluation of VAE robustness
import matplotlib

matplotlib.use('Agg')
import argparse
import scipy.optimize
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from model import VAE, VAE_vanila
import torch
import torch.nn as nn
from prepareinput import prepare_data_loader
from attackutils import sample_d_ball
import pytorch_msssim

loss = torch.nn.MSELoss()
attack_losses = []
recon_dist = []
norm_recon_dist = []
mse = []
msssim = []
inp_dist = []
inp_msssim = []
distances = []
out_msssim = []
rmsssim = []
amsssim = []


def max_damage_optimize_noise(model, image, maximum_noise_norm, img_size, nc, d_ball_init=True, scale=False):
    if d_ball_init:
        initial_noise = sample_d_ball(img_size * img_size * nc, maximum_noise_norm).reshape(
            (nc, img_size, img_size)).astype(np.float32)
    else:
        initial_noise = np.random.uniform(-1e-8, 1e-8, size=(nc, img_size, img_size)).astype(
            np.float32)

    adversarial_losses = []

    def fmin_func(noise):
        loss, gradient = model.module.eval_max_damage_attack(image, noise, maximum_noise_norm, scale=scale)
        adversarial_losses.append(loss.detach().cpu().numpy())
        return float(loss.data.cpu().detach().numpy()), gradient.data.cpu().detach().numpy().flatten().astype(
            np.float64)

    # BB: Bounds on the noise to ensure pixel values remain in interval [0, 1]
    lower_limit = -image.data.numpy().flatten()
    upper_limit = (1.0 - image.data.numpy().flatten())

    bounds = zip(lower_limit, upper_limit)
    bounds = [sorted(y) for y in bounds]

    # BB: Optimizer to find adversarial noise
    noise, attackloss, _ = scipy.optimize.fmin_l_bfgs_b(fmin_func,
                                                        x0=initial_noise,
                                                        bounds=bounds,
                                                        m=100,
                                                        factr=10,
                                                        pgtol=1e-20)
    attack_losses.append(attackloss)
    return (torch.tensor(noise).view(1, nc, img_size,
                                     img_size)).float(), adversarial_losses, attack_losses


def max_damage_attack(model_exp_dir, model, iterator, maximum_noise_norm, num_images, img_size, dataset, modeltype, nc,
                      d_ball_init=True):
    sample = next(iter(iterator))
    attack_sample = (sample[0][:num_images], sample[1][:num_images])

    for i in range(num_images):
        image_index = 65
        original_image = attack_sample[0][image_index]
        noise, losses, attacklosses = max_damage_optimize_noise(model, original_image, maximum_noise_norm, img_size, nc,
                                                                d_ball_init=d_ball_init, scale=False)
        if noise.norm(p=2) > maximum_noise_norm:
            noise = maximum_noise_norm * noise.div(noise.norm(p=2))
        noisy_image = original_image + noise.view(nc, img_size,
                                                  img_size)

        if modeltype == "VAE":
            clean_reconstruction, _, _ = model(original_image[None, :].cuda())
            noisy_reconstruction, _, _ = model(noisy_image.cuda())
        else:
            clean_reconstruction, _, _, _ = model(original_image[None, :].cuda())
            noisy_reconstruction, _, _, _ = model(noisy_image[None, :].cuda())

        reshaped_noisy_reconstruction = noisy_reconstruction.view(1, nc, img_size,
                                                                  img_size)
        reshaped_clean_reconstruction = clean_reconstruction.view(1, nc, img_size,
                                                                  img_size)

        noise = noise.view(1, nc, img_size,
                           img_size)
        if dataset == "MNIST" or dataset == "FASHIONMNIST":
            image_compilation = torch.cat((original_image.unsqueeze(0).cuda(),
                                           reshaped_clean_reconstruction.cuda(),
                                           #noise.cuda(),
                                           noisy_image.unsqueeze(0).cuda(),
                                           reshaped_noisy_reconstruction.cuda()), dim=-1)

        else:
            image_compilation = torch.cat((original_image.permute(1, 2, 0).cuda(),
                                           clean_reconstruction.squeeze().permute(1, 2, 0).cuda(),
                                           #noise.squeeze().permute(1, 2, 0).cuda(),
                                           noisy_image.squeeze().permute(1, 2, 0).cuda(),
                                           noisy_reconstruction.squeeze().permute(1, 2, 0).cuda()), dim=-2)

        # Plotting
        plt.figure(figsize=(9, 3))
        if dataset == "MNIST" or dataset == "FASHIONMNIST":
            plt.imshow(image_compilation.detach().cpu().squeeze().numpy(), vmin=0, vmax=1, cmap="gray")
        else:
            plt.imshow((image_compilation.detach().cpu().numpy() * 255).astype(np.uint8))
        plt.axis('off')
        plotting_dir = model_exp_dir + "/maxdamageattack/"
        os.makedirs(plotting_dir, exist_ok=True)
        plt.savefig(plotting_dir + "attack_{}_noise_coefficient_{}.png".format(i + 1,
                                                                                      maximum_noise_norm), dpi=300, bbox_inches='tight')

        msssim.append(
            pytorch_msssim.msssim(original_image.unsqueeze(0), noisy_image.unsqueeze(0), window_size=6,
                                  normalize='relu').detach().cpu().numpy())
        out_msssim.append(
            pytorch_msssim.msssim(reshaped_noisy_reconstruction, reshaped_clean_reconstruction, window_size=6,
                                  normalize='relu').detach().cpu().numpy())
        distances.append(
            ((reshaped_noisy_reconstruction.flatten() - reshaped_clean_reconstruction.flatten()).norm(
                p=2)).detach().cpu().numpy())
        original_image = original_image.cuda()
        noisy_image = noisy_image.cuda()
        rmsssim.append(pytorch_msssim.msssim(original_image.unsqueeze(0), reshaped_clean_reconstruction, window_size=6,
                                             normalize='relu').detach().cpu().numpy())
        amsssim.append(pytorch_msssim.msssim(noisy_image.unsqueeze(0), reshaped_noisy_reconstruction, window_size=6,
                                             normalize='relu').detach().cpu().numpy())



    result_txt = plotting_dir + "/outputspaceattack_metrics.txt"
    with open(result_txt, "a+") as myfile:
        myfile.write(str("maximum_noise_norm - ") + str(maximum_noise_norm) + "\n")
        myfile.write(str("MSSSIM - ") + str(np.mean(msssim)) + "\n")
        myfile.write(str("out MSSSIM - ") + str(np.mean(out_msssim)) + "\n")
        myfile.write(str("recon_dist - ") + str(np.mean(distances)) + "\n")
        myfile.write(str("attack_losses - ") + str(np.mean(attacklosses)) + "\n")


def max_damage_attack_model(opt):
    dataset = opt['dataset']
    model_exp_dir = opt['modelpath']
    modelpath = model_exp_dir
    latentsize = opt['latentsize']
    image_channel = opt['image_channel']
    img_size = opt['img_size']
    modeltype = opt['modeltype']
    datadir = opt['datadir']
    if modeltype == "VAE":
        model = VAE_vanila(dataset=dataset, nc=image_channel, ndf=128, nef=128, nz=latentsize, isize=img_size)
    else:
        model = VAE(dataset=dataset, nc=image_channel, ndf=128, nef=128, nz=latentsize, isize=img_size,
                    is_train=True, latent_noise_scale=0.004)
    print("model loaded!")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(modelpath))

    if opt['data']['cuda']:
        print('Using CUDA')
        model.cuda()

    # get data

    trainloader, testloader, valloader = prepare_data_loader(datadir, 128, 128, 128, dataset=dataset)

    model.eval()

    # Inspect r-robustness probability degradation w.r.t. norm of max damage attacks and model
    max_damage_attack(model_exp_dir, model, testloader,
                      opt['maximum_noise_norm'],
                      opt['num_max_damage_images'], opt['img_size'], dataset, modeltype, image_channel,
                      d_ball_init=opt['d_ball_init'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Output attack trained VAE')
    parser.add_argument('--modelpath', type=str, metavar='model path',
                        help="location of pretrained model weights to evaluate")
    parser.add_argument('--modeltype', type=str, metavar='model type',
                        help="type of pretrained model to evaluate")
    parser.add_argument('--datadir', type=str, metavar='data directory',
                        help="dataset to evaluate")
    parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")
    parser.add_argument('--num_max_damage_images', type=int, default=50, help='number of images to perform attack on')
    parser.add_argument('--maximum_noise_norm', type=float, default=5.0,
                        help='maximal norm of noise in max damage attack')
    parser.add_argument('--d_ball_init', type=bool, default=True,
                        help='whether attack noise should be initialized from random point in d-ball around image (True/False)')
    parser.add_argument('--num_random_inits', type=int, default=5,
                        help='how many random initializations of attack noise to use (int)')
    parser.add_argument('--dataset', type=str,
                        help="dataset name to use")
    parser.add_argument('--savedir', type=str,
                        help="path to save results")
    parser.add_argument('--latentsize', type=int, default=10,
                        help='size of the ls of the model to perform latent space attack on')
    parser.add_argument('--img_size', type=int, default=32, help='size of images to perform attack on')
    parser.add_argument('--image_channel', type=int, default=1,
                        help='number of channels in images to perform attack on')

    args = vars(parser.parse_args())

    print("Args: {}".format(args))

    opt = {}
    for k, v in args.items():
        cur = opt
        tokens = k.split('.')
        for token in tokens[:-1]:
            if token not in cur:
                cur[token] = {}
            cur = cur[token]
        cur[tokens[-1]] = v

    max_damage_attack_model(opt)
