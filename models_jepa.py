import torch
import torch.nn as nn
import torch.nn.functional as F
from util.pos_embed import get_1d_sincos_pos_embed
from util.masking import gather_tokens

try:
    import ptwt
except ImportError:
    ptwt = None


class WaveletTokenEmbed1D(nn.Module):
    """Wavelet tokenization for variable-length wearable time-series.

    Each channel is decomposed with a discrete wavelet transform; every level is
    pooled into ``tokens_per_level`` tokens carrying a (mean, energy) feature pair,
    yielding ``(level + 1) * tokens_per_level * num_channels`` tokens.
    """

    def __init__(self, num_channels=6, embed_dim=384, wavelet='db4', level=4, tokens_per_level=2):
        super().__init__()
        if ptwt is None:
            raise ImportError('ptwt is required for wavelet tokenization')
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.wavelet = wavelet
        self.level = level
        self.tokens_per_level = tokens_per_level
        self.num_levels = level + 1
        self.num_patches = self.num_levels * tokens_per_level * num_channels

        self.value_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.level_embed = nn.Parameter(torch.zeros(1, self.num_levels, 1, 1, embed_dim))
        self.channel_embed = nn.Parameter(torch.zeros(1, num_channels, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, tokens_per_level, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

    def _match_channels(self, x):
        channels = x.shape[1]
        if channels == self.num_channels:
            return x
        if channels > self.num_channels:
            return x[:, : self.num_channels]
        pad = torch.zeros(
            x.shape[0],
            self.num_channels - channels,
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, pad], dim=1)

    def _iter_features(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(
                f'Expected input shape [B, C, T] or [B, 1, C, T], got {tuple(x.shape)}'
            )

        x = self._match_channels(x)
        coeffs = ptwt.wavedec(x, self.wavelet, mode='reflect', level=self.level, axis=-1)

        for coeff in coeffs:
            pooled_mean = F.adaptive_avg_pool1d(coeff, self.tokens_per_level)
            pooled_energy = torch.sqrt(
                F.adaptive_avg_pool1d(coeff.square(), self.tokens_per_level) + 1e-6
            )
            yield torch.stack((pooled_mean, pooled_energy), dim=-1)

    def extract_features(self, x):
        """Parameter-free wavelet statistics, shareable by the two JEPA encoders."""
        return list(self._iter_features(x))

    def forward(self, x, *, features=None):
        if features is None:
            features = self._iter_features(x)
        # Keep the original per-level projection/reduction order for training
        # reproducibility; combining levels changes floating-point gradients.
        tokens = []
        for level_idx, values in enumerate(features):
            token = self.value_proj(values)
            token = token + self.level_embed[:, level_idx] + self.channel_embed + self.time_embed
            tokens.append(token.reshape(token.shape[0], -1, self.embed_dim))
        return torch.cat(tokens, dim=1)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        normalized = self.norm1(x)
        x = x + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Wavelet-tokenized transformer encoder."""

    def __init__(
        self,
        num_channels=6,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        wavelet='db4',
        wavelet_level=4,
        wavelet_tokens_per_level=2,
        **unused,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = WaveletTokenEmbed1D(
            num_channels=num_channels,
            embed_dim=embed_dim,
            wavelet=wavelet,
            level=wavelet_level,
            tokens_per_level=wavelet_tokens_per_level,
        )
        self.num_patches = self.patch_embed.num_patches

        self.register_buffer(
            'pos_embed',
            torch.zeros(1, self.num_patches, embed_dim),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.embed_dim, self.num_patches)
        self.pos_embed.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None, *, wavelet_features=None):
        x = self.patch_embed(x, features=wavelet_features)  # [B, tokens, D]

        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = (
                torch.from_numpy(get_1d_sincos_pos_embed(self.embed_dim, x.shape[1]))
                .to(device=x.device, dtype=x.dtype)
                .unsqueeze(0)
            )
        else:
            pos_embed = self.pos_embed.to(device=x.device, dtype=x.dtype)

        x = x + pos_embed

        if mask is not None:  # [B, K] indices of the visible tokens
            x = gather_tokens(x, mask)

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)


class Predictor(nn.Module):
    """Predicts masked target-token representations from visible context tokens."""

    def __init__(
        self,
        num_patches,
        encoder_embed_dim=384,
        predictor_embed_dim=192,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.predictor_embed_dim = predictor_embed_dim
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [Block(predictor_embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.predictor_embed_dim, self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, context_features, context_mask, target_mask):
        # Masks are [B, K] token indices, so one batched gather positions every
        # context token and one concatenation appends the target queries.
        context = self.predictor_embed(context_features)
        positions = self.pos_embed.expand(context.shape[0], -1, -1)
        context = context + gather_tokens(positions, context_mask)
        targets = self.mask_token + gather_tokens(positions, target_mask)
        x = torch.cat((context, targets), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.predictor_proj(self.norm(x))
        return x[:, context.shape[1] :]


class SensorJEPA(nn.Module):
    """Wavelet W-JEPA: context encoder + latent predictor (target encoder is an EMA copy)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.context_encoder = VisionTransformer(
            num_channels=config['num_channels'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            mlp_ratio=config.get('mlp_ratio', 4.0),
            wavelet=config.get('wavelet', 'db4'),
            wavelet_level=config.get('wavelet_level', 4),
            wavelet_tokens_per_level=config.get('wavelet_tokens_per_level', 2),
        )
        self.predictor = Predictor(
            num_patches=self.context_encoder.num_patches,
            encoder_embed_dim=config['embed_dim'],
            predictor_embed_dim=config['predictor_embed_dim'],
            depth=config['predictor_depth'],
            num_heads=config['predictor_num_heads'],
        )

    def forward(self, x, context_mask, target_mask):
        context_features = self.context_encoder(x, context_mask)
        return self.predictor(context_features, context_mask, target_mask)


def sensor_jepa_base(**dataset_config):
    from config import BASE_MODEL_CONFIG

    config = {**BASE_MODEL_CONFIG, **dataset_config}
    return SensorJEPA(config)
