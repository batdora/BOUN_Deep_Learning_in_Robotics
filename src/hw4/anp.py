"""Attentive CNMP: cross-attention aggregator in place of mean pooling.

This keeps the same public interface as the stock `CNP` from `homework4.py`
(`forward`, `nll_loss` with identical signatures) so the training and
evaluation scripts can use it via a simple `--model-type` flag.

Design:
    Encoder       : (context_x, context_y) -> value per context point
    Key  / Query  : small linear maps from x only
    Attention     : multi-head, each target query attends over context keys
    Decoder       : concat(attended_r_j, target_x_j) -> (mean, logstd)

The std head is kept identical to `CNP`: `softplus(logstd) + min_std`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, out_dim, hidden_size, num_hidden_layers):
    """Stock MLP: ReLU hidden, linear head. Mirrors `CNP` layer counts."""
    layers = [nn.Linear(in_dim, hidden_size), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, out_dim))
    return nn.Sequential(*layers)


class AttentiveCNP(nn.Module):
    def __init__(self, in_shape, hidden_size, num_hidden_layers,
                 num_heads=4, min_std=0.1):
        super().__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]
        self.hidden_size = hidden_size
        self.min_std = min_std

        assert hidden_size % num_heads == 0, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"

        # Value encoder: (x, y) per context point -> hidden_size
        self.encoder = _mlp(self.d_x + self.d_y, hidden_size,
                            hidden_size, num_hidden_layers)

        # Key / query projections operate on x only -- "attend by time/height"
        self.to_key = nn.Linear(self.d_x, hidden_size)
        self.to_query = nn.Linear(self.d_x, hidden_size)

        self.attn = nn.MultiheadAttention(hidden_size, num_heads,
                                          batch_first=True)

        # Decoder: concat(attended, target_x) -> (mean, logstd)
        self.decoder = _mlp(hidden_size + self.d_x, 2 * self.d_y,
                            hidden_size, num_hidden_layers)

    def forward(self, observation, target, observation_mask=None):
        """
        Parameters
        ----------
        observation : (B, N_ctx, d_x + d_y)
        target      : (B, N_tgt, d_x)
        observation_mask : (B, N_ctx) or None
            1 for valid context entries, 0 for padding.
        """
        ctx_x = observation[..., :self.d_x]
        values = self.encoder(observation)  # (B, N_ctx, H)
        keys = self.to_key(ctx_x)            # (B, N_ctx, H)
        queries = self.to_query(target)      # (B, N_tgt, H)

        key_padding_mask = None
        if observation_mask is not None:
            # nn.MultiheadAttention: True = ignore
            key_padding_mask = ~observation_mask.bool()

        attended, _ = self.attn(queries, keys, values,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)                # (B, N_tgt, H)

        h_cat = torch.cat([attended, target], dim=-1)
        out = self.decoder(h_cat)                                  # (B, N_tgt, 2*d_y)
        mean = out[..., :self.d_y]
        logstd = out[..., self.d_y:]
        std = F.softplus(logstd) + self.min_std
        return mean, std

    def nll_loss(self, observation, target, target_truth,
                 observation_mask=None, target_mask=None):
        mean, std = self.forward(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss
