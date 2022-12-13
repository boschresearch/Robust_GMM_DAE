"Model definition"
"code adapted but modified from https://github.com/boschresearch/GMM_DAE"
# ----------------------------------------------------------------------------
import torch
import torch.utils.data
from torch import nn

from torch.nn import functional as F


class MNISTEncoderFC(nn.Module):
    def __init__(self, nc, nef, nz, isize, device):
        super(MNISTEncoderFC, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)
        self.encoder = nn.Sequential(

            nn.Linear(32 * 32 * 1, 200),
            nn.ReLU(True),

            nn.Linear(200, 200),
            nn.ReLU(True),

            nn.Linear(200, 200),
            nn.ReLU(True),

            nn.Linear(200, 200),
            nn.ReLU(True)

        )
        self.fc1 = nn.Linear(200, nz)
        self.fc2 = nn.Linear(200, nz)

    def forward(self, inputs):
        # Batch size
        batch_size = inputs.size(0)
        # Reshape
        inputs = inputs.view(batch_size, -1)
        latent_z = self.encoder(inputs)
        mu = self.fc1(latent_z)
        return mu


class MNISTDecoderFC(nn.Module):
    def __init__(self, nc, ndf, nz, isize):
        super(MNISTDecoderFC, self).__init__()
        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(
            nn.Linear(200, 1 * 32 * 32),
            nn.Sigmoid(),

        )
        self.decoder_conv = nn.Sequential(
            nn.Linear(nz, 200),
            nn.ReLU(True),

            nn.Linear(200, 200),
            nn.ReLU(True),

            nn.Linear(200, 200),
            nn.ReLU(True),

            nn.Linear(200, 200),
            nn.ReLU(True)
        )

    def forward(self, input):
        output = self.decoder_conv(input)
        output = self.fc1(output)
        output = output.view(input.shape[0], (32 * 32 * 1))
        output = torch.reshape(output, (input.shape[0], 1, 32, 32))
        return output

class SVHNEncoder(nn.Module):
    def __init__(self, nc, nef, nz, isize, device):
        super(SVHNEncoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)
        self.encoder = nn.Sequential(

            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.BatchNorm2d(nef),
            nn.ReLU(True),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 2),
            nn.ReLU(True),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 4),
            nn.ReLU(True),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.BatchNorm2d(nef * 8),
            nn.ReLU(True)

        )
        out_size = isize // 16
        self.fc1 = nn.Linear(nef * 8 * out_size * out_size, nz)

    def forward(self, inputs):
        # Batch size
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        hidden = hidden.view(batch_size, -1)
        latent_z = self.fc1(hidden)
        return latent_z


class SVHNDecoder(nn.Module):
    def __init__(self, nc, ndf, nz, isize):
        super(SVHNDecoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(
            nn.Linear(nz, 2 * 2 * 1024),
            nn.ReLU(True)
        )
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.conv1 = nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1)
        self.decoder_conv = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),

            self.conv2,
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),

            self.conv3,
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),

            self.conv4,
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.fc1(input)
        input = input.view(input.size(0), 1024, 2, 2)
        output = self.decoder_conv(input)
        return output


class CELEBEncoder(nn.Module):
    def __init__(self, nc, nef, nz, isize, device):
        super(CELEBEncoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)

        self.encoder = nn.Sequential(

            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef),

            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 2),

            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 4),

            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 8),
        )
        out_size = isize // 16
        self.fc1 = nn.Linear(nef * 8 * out_size * out_size, nz)

    def forward(self, inputs):
        # Batch size
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        hidden = hidden.view(batch_size, -1)
        latent_z = self.fc1(hidden)
        return latent_z


class CELEBDecoder(nn.Module):
    def __init__(self, nc, ndf, nz, isize):
        super(CELEBDecoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(
            nn.Linear(nz, 4 * 4 * 1024),
            nn.ReLU(True),

        )
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.conv1 = nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1)
        self.decoder_conv = nn.Sequential(
            self.conv1,
            nn.ReLU(True),
            nn.BatchNorm2d(ndf * 4),

            self.conv2,
            nn.ReLU(True),
            nn.BatchNorm2d(ndf * 2),

            self.conv3,
            nn.ReLU(True),
            nn.BatchNorm2d(ndf),

            self.conv4,
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.fc1(input)
        input = input.view(input.size(0), 1024, 4, 4)
        output = self.decoder_conv(input)
        return output

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class VAE(nn.Module):
    def __init__(self, dataset="MNIST", nc=1, ndf=32, nef=32, nz=16, isize=128, latent_noise_scale=0,
                 gmm_centers=None, gmm_std=None,
                 device=torch.device("cuda:0"), is_train=True):
        super(VAE, self).__init__()
        self.nz = nz
        self.is_train = is_train
        self.latent_noise_scale = latent_noise_scale
        self.isize = isize
        self.gmm_centers = gmm_centers
        self.gmm_std = gmm_std
        self.nc = nc

        if dataset == "MNIST" or dataset == "FASHIONMNIST":
            # Encoder
            self.encoder = MNISTEncoderFC(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = MNISTDecoderFC(nc=nc, ndf=ndf, nz=nz, isize=isize)
        elif dataset == "SVHN":
            # Encoder
            self.encoder = SVHNEncoder(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = SVHNDecoder(nc=nc, ndf=ndf, nz=nz, isize=isize)
        elif dataset == "CELEB":
            # Encoder
            self.encoder = CELEBEncoder(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = CELEBDecoder(nc=nc, ndf=ndf, nz=nz, isize=isize)

    def forward(self, images, adv_images=None):
        z = self.encode(images)
        if adv_images is not None:
            z_adv = self.encode(adv_images)
            adv_recon = self.decode(z_adv)
        else:
            z_adv = None
            adv_recon = None
        if self.is_train:
            z_noise = self.latent_noise_scale * torch.randn((images.size(0), self.nz),
                                                            device=z.device)
        else:
            z_noise = 0.0

        return self.decode(z + z_noise), adv_recon, z, z_adv

    def encode(self, images):
        return self.encoder(images)

    def decode(self, z):
        return self.decoder(z)

    def craft_adversarial_samples(self, images, epsilon, alpha):

        # Normal images' latent
        latent_images = self.encode(images)
        # FGSM attack to create adversarial samples based on latent loss
        delta = torch.zeros_like(images).uniform_(-epsilon, epsilon).cuda()
        delta.requires_grad = True
        advlatents = self.encode(images + delta)
        latentloss = F.mse_loss(latent_images, advlatents)
        # latentloss = (latent_images - advlatents).norm(p=2)
        latentloss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1 - images, delta.data), 0 - images)
        delta = delta.detach()
        adv_images = torch.clamp(images + delta, 0, 1)
        return adv_images
