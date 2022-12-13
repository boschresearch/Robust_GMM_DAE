"from https://github.com/FabianBarrett/Lipschitz_VAEs/blob/master/lnets/tasks/vae/mains/latent_space_attack.py - no modifications"
import numpy as np


def sample_d_ball(d, radius):
    u = np.random.randn(d)
    norm = (u ** 2).sum() ** 0.5
    r = np.random.random() ** (1.0 / d)
    sample = (r * u) / norm
    scaling = np.random.uniform(1e-8, radius)
    return scaling * sample


def get_target_image(batch, input_class, input_index, num_images):
    image_counter = 0
    image_index = input_index + 1
    while image_counter < (num_images - 1):
        image_index %= num_images
        if batch[1][image_index] != input_class:
            return batch[0][image_index], batch[1][image_index]
        image_counter += 1
        image_index += 1
    raise RuntimeError("No appropriate target image found.")


def get_target_imageclassifier(data, target, input_class, input_index, num_images):
    image_counter = 0
    image_index = input_index + 1
    while image_counter < (num_images - 1):
        image_index %= num_images
        if target[image_index] != input_class:
            return data[image_index], target[image_index]
        image_counter += 1
        image_index += 1
    raise RuntimeError("No appropriate target image found.")


def get_target_image_celeb(batch, input_index, num_images):
    image_index = np.random.randint(0, num_images)
    # image_index = 75
    while input_index == image_index:
        image_index = np.random.randint(0, num_images)

    return batch[0][image_index], batch[1][image_index]
