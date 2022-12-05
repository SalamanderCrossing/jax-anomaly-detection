import jax.numpy as jnp
import flax.linen as nn
from .multihead_attention import MultiheadAttention


class EncoderBlock(nn.Module):
    input_dim: int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        # Attention layer
        self.self_attn = MultiheadAttention(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim),
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        # Attention part
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x
