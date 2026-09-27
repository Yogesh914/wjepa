import numpy as np


def get_1d_sincos_pos_embed(embed_dim, num_patches):
    grid_t = np.arange(num_patches, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)
