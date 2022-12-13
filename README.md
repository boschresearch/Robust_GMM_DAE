# Trading off Image Quality for Robustness is not Necessary with Deterministic Autoencoders PyTorch 

PyTorch implementation of the NeurIPS 2022 paper "Trading off Image Quality for Robustness is not Necessary with Deterministic Autoencoders". The code allows the users to
reproduce and extend the results reported in the paper. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication. It will neither be
maintained nor monitored in any way.

## Setup.

1. Create a conda virtual environment
2. Clone the repository
3. Activate the environment and run 
 ```bash
cd Robust_GMM_DAE
pip install requirements.txt
```
## Dataset

The provided implementation is tested on MNIST, FASHION MNIST, SVHN and CELEBA images. We follow the same procedure in loading the data as in [here.](https://github.com/boschresearch/GMM_DAE)
  
### Usage

To train the model, clone the repository and then run

```bash
python train.py <dataset_name> eg: MNIST, FASHIONMNIST, SVHN or CELEB
```

To evaluate the robustness of the model against VAE attacks, we consider two types of attack, latent space and maximum damage attack. In the attacks directory, run

```bash
python attacks/latentspaceattacks.py  --dataset=<dataset_name> --num_images=<number_of_images> --datadir=<path_to_the_dataset> --modelpath=<path_to_the_trainedmodel> --maximum_noise_norm=<1,3,5>
python attacks/outputattack.py --dataset=<dataset_name> --num_max_damage_images=<number_of_images> --datadir=<path_to_the_dataset> --modelpath=<path_to_the_trainedmodel> --maximum_noise_norm=<1,3,5>
```

For FID computation we used the github repo [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
## License

Robust_GMM_DAE is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

