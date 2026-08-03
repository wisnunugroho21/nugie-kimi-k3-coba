import math

import jax
import jax.numpy as jnp
from flax import nnx


class MultiHeadLatentAttentionNoRoPE(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_head_dim: int,
        v_head_dim: int,
        q_latent_dim: int,
        kv_latent_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim

        self.w_dq = nnx.Linear(d_model, q_latent_dim, use_bias=False, rngs=rngs)
        self.w_uq = nnx.Linear(
            q_latent_dim, num_heads * q_head_dim, use_bias=False, rngs=rngs
        )

        self.w_dkv = nnx.Linear(d_model, kv_latent_dim, use_bias=False, rngs=rngs)
        self.w_uk = nnx.Linear(
            kv_latent_dim, num_heads * q_head_dim, use_bias=False, rngs=rngs
        )
        self.w_uv = nnx.Linear(
            kv_latent_dim, num_heads * v_head_dim, use_bias=False, rngs=rngs
        )

        self.w_o = nnx.Linear(
            num_heads * v_head_dim, d_model, use_bias=False, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        c_q = self.w_dq(x)
        c_kv = self.w_dkv(x)

        q = self.w_uq(c_q)
        k = self.w_uk(c_kv)
        v = self.w_uv(c_kv)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.q_head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.q_head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.v_head_dim)

        q = q.swapaxes(2, 1)
        k = k.swapaxes(2, 1)
        v = v.swapaxes(2, 1)

        scale = 1.0 / math.sqrt(self.q_head_dim)
        attn_scores = q @ k.swapaxes(-1, -2) * scale

        if mask is not None:
            attn_scores = jnp.where(mask, attn_scores, -1e9)

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        out = attn_weights @ v
        out = out.swapaxes(2, 1)
        out = out.reshape(batch_size, seq_len, -1)

        return self.w_o(out)
