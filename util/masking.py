import torch


class RandomTubeMasking1D:
    """Mask entire (wavelet band, channel) groups of two temporal tokens.

    The tokenizer flattens [band, channel, temporal token], so contiguous
    pairs are exactly the two tokens from one band of one input channel.
    Floor the requested fraction in group space: 60 tokens at 0.65 means
    19 of 30 pairs (38 target tokens), with no partially visible pair.
    """

    def __init__(self, num_patches, mask_ratio=0.65, group_size=2):
        if group_size != 2:
            raise ValueError('Masking groups are the two temporal tokens of one band/channel')
        if num_patches <= 0 or num_patches % group_size:
            raise ValueError('num_patches must contain complete two-token groups')
        if not 0 < mask_ratio < 1:
            raise ValueError('mask_ratio must be between zero and one')
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.group_size = group_size
        self.num_groups = num_patches // group_size
        self.num_masked_groups = int(self.num_groups * mask_ratio)
        if not 0 < self.num_masked_groups < self.num_groups:
            raise ValueError('Both target and context must contain a complete group')
        self.num_masked = self.num_masked_groups * group_size

    def sample_indices(self, batch_size, device='cuda'):
        """Sample groups independently per example and retain original token order."""
        groups = torch.stack([
            torch.randperm(self.num_groups, device=device) for _ in range(batch_size)
        ])
        offsets = torch.arange(self.group_size, device=device)

        def expand(selected):
            ordered = selected.sort(dim=1).values
            return (ordered.unsqueeze(-1) * self.group_size + offsets).flatten(1)

        target = expand(groups[:, :self.num_masked_groups])
        context = expand(groups[:, self.num_masked_groups:])
        return context, target


def gather_tokens(tokens, indices):
    """Gather ordered tokens from [B, N, D] with [B, K] integer indices."""
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
