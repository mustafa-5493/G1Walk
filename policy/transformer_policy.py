import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

# Walter's design: Transformer encoder over 8 frames, 256 hidden, 3 layers

class TransformerPolicy(nn.Module):
    """
    Transformer-based actor-critic for G1 locomotion.

    Design (Walter):
    - 8-frame observation history
    - Transformer encoder: 256 embed dim, 4 heads, 3 layers
    - Separate MLP critic (current obs only)
    - Orthogonal initialization
    """

    def __init__(self, obs_dim, act_dim, history_len=8,
                 embed_dim=256, n_heads=4, n_layers=3, ffn_dim=512):
        super().__init__()
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.history_len = history_len
        self.embed_dim   = embed_dim

        # input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
        )

        # learnable positional embedding
        self.pos_emb = nn.Parameter(
            torch.zeros(1, history_len, embed_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=0.0,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # actor head
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim * history_len, 256),
            nn.ELU(),
            nn.Linear(256, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # critic — uses current obs only (last frame)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 256),    nn.ELU(),
            nn.Linear(256, 256),    nn.ELU(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # small init for actor output — start near zero actions
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)

    def forward(self, obs_history):
        """
        obs_history: (batch, history_len, obs_dim)
        returns: mean, std, value
        """
        x = self.input_proj(obs_history) + self.pos_emb
        x = self.transformer(x)
        x = x.flatten(1)
        mean  = self.actor_head(x)
        std   = self.log_std.exp().expand_as(mean)
        value = self.critic(obs_history[:, -1, :])
        return mean, std, value

    def get_action(self, obs_history):
        mean, std, value = self.forward(obs_history)
        dist     = Normal(mean, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, obs_history, action):
        mean, std, value = self.forward(obs_history)
        dist     = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, value.squeeze(-1)
