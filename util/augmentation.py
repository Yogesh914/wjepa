import numpy as np


def jittering(sample, noise_level=0.05):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=sample.shape)
    return sample + noise


def scaling(sample, scale_factor=0.1):
    sample *= np.random.normal(loc=1.0, scale=scale_factor)
    return sample


def finetune_aug(x, noise_level=0.03, scale_factor=0.05):
    """Light train-only augmentation used for downstream fine-tuning."""
    x = x.astype(np.float32).copy()
    x = jittering(x, noise_level=noise_level)
    x = scaling(x, scale_factor=scale_factor)
    return x
