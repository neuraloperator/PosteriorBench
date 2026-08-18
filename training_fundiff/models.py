from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
import flax.linen as nn


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: jnp.ndarray) -> jnp.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("Sine-cosine embeddings require an even dimension")
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> jnp.ndarray:
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim,
            jnp.arange(length, dtype=jnp.float32),
        ),
        0,
    )


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]) -> jnp.ndarray:
    def from_grid(embed_dim: int, grid: jnp.ndarray) -> jnp.ndarray:
        if embed_dim % 2 != 0:
            raise ValueError("2D sine-cosine embeddings require an even dimension")
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return jnp.concatenate([emb_h, emb_w], axis=1)

    grid_h = jnp.arange(grid_size[0], dtype=jnp.float32)
    grid_w = jnp.arange(grid_size[1], dtype=jnp.float32)
    grid = jnp.meshgrid(grid_h, grid_w, indexing="ij")
    grid = jnp.stack(grid, axis=0).reshape([2, 1, grid_size[0], grid_size[1]])
    return jnp.expand_dims(from_grid(embed_dim, grid), 0)


class PatchEmbed(nn.Module):
    patch_size: tuple[int, int] = (16, 16)
    emb_dim: int = 256
    use_norm: bool = False
    kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        batch, height, width, _ = inputs.shape
        if height % self.patch_size[0] or width % self.patch_size[1]:
            raise ValueError(
                f"Input shape {(height, width)} is not divisible by patch size "
                f"{self.patch_size}"
            )
        x = nn.Conv(
            self.emb_dim,
            self.patch_size,
            self.patch_size,
            kernel_init=self.kernel_init,
            name="proj",
        )(inputs)
        x = jnp.reshape(x, (batch, -1, self.emb_dim))
        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=1e-5)(x)
        return x


class MlpBlock(nn.Module):
    dim: int
    out_dim: int
    kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)
        x = nn.gelu(x)
        return nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)


class SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.emb_dim,
        )(x, x)
        x = x + inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
        return x + y


class CrossAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs: jnp.ndarray, kv_inputs: jnp.ndarray) -> jnp.ndarray:
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.emb_dim,
        )(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
        return x + y


class PerceiverBlock(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int = 8
    num_latents: int = 64
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        latents = self.param(
            "latents",
            nn.initializers.normal(),
            (self.num_latents, self.emb_dim),
        )
        latents = repeat(latents, "l d -> b l d", b=x.shape[0])
        for _ in range(self.depth):
            latents = CrossAttnBlock(
                self.num_heads,
                self.emb_dim,
                self.mlp_ratio,
                self.layer_norm_eps,
            )(latents, x)
        return nn.LayerNorm(epsilon=self.layer_norm_eps)(latents)


class Encoder(nn.Module):
    patch_size: tuple[int, int]
    grid_size: tuple[int, int]
    emb_dim: int
    num_latents: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, height, width, _ = x.shape
        x = PatchEmbed(self.patch_size, self.emb_dim)(x)
        pos_emb = self.variable(
            "pos_emb",
            "enc_pos_emb",
            get_2d_sincos_pos_embed,
            self.emb_dim,
            (
                self.grid_size[0] // self.patch_size[0],
                self.grid_size[1] // self.patch_size[1],
            ),
        )
        pos_emb_interp = pos_emb.value.reshape(
            1,
            self.grid_size[0] // self.patch_size[0],
            self.grid_size[1] // self.patch_size[1],
            self.emb_dim,
        )
        pos_emb_interp = jax.image.resize(
            pos_emb_interp,
            (
                1,
                height // self.patch_size[0],
                width // self.patch_size[1],
                self.emb_dim,
            ),
            method="bilinear",
        )
        x = x + rearrange(pos_emb_interp, "b h w d -> b (h w) d")
        x = PerceiverBlock(
            emb_dim=self.emb_dim,
            depth=2,
            num_heads=self.num_heads,
            num_latents=self.num_latents,
        )(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads,
                self.emb_dim,
                self.mlp_ratio,
                self.layer_norm_eps,
            )(x)
        return nn.LayerNorm(epsilon=self.layer_norm_eps)(x)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel",
            nn.initializers.normal(self.embed_scale),
            (x.shape[-1], self.embed_dim // 2),
        )
        y = jnp.dot(x, kernel)
        return jnp.concatenate([jnp.cos(y), jnp.sin(y)], axis=-1)


class PeriodEmbs(nn.Module):
    period: tuple[float, ...]
    axis: tuple[int, ...]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        values = []
        for index, xi in enumerate(x):
            if index in self.axis:
                period = self.period[self.axis.index(index)]
                values.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                values.append(xi)
        return jnp.hstack(values)


class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
            x = nn.gelu(x)
        return nn.Dense(features=self.out_dim)(x)


class Decoder(nn.Module):
    fourier_freq: float = 1.0
    period: bool = False
    dec_depth: int = 2
    dec_num_heads: int = 8
    dec_emb_dim: int = 256
    mlp_ratio: int = 1
    out_dim: int = 1
    num_mlp_layers: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, z: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        batch, _, _ = z.shape
        if self.period:
            coords = PeriodEmbs(period=(2 * jnp.pi, 2 * jnp.pi), axis=(0, 1))(coords)
        coords = FourierEmbs(
            embed_scale=self.fourier_freq,
            embed_dim=self.dec_emb_dim,
        )(coords)
        coords = repeat(coords, "d -> b n d", n=1, b=batch)
        x = nn.Dense(self.dec_emb_dim)(z)
        for _ in range(self.dec_depth):
            coords = CrossAttnBlock(
                num_heads=self.dec_num_heads,
                emb_dim=self.dec_emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
            )(coords, x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)
        return Mlp(
            num_layers=self.num_mlp_layers,
            hidden_dim=self.dec_emb_dim,
            out_dim=self.out_dim,
        )(x)


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return x * (1 + scale[:, None]) + shift[:, None]


class TimestepEmbedder(nn.Module):
    emb_dim: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        x = self.timestep_embedding(t)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        return nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02))(x)

    def timestep_embedding(self, t: jnp.ndarray, max_period: int = 10000) -> jnp.ndarray:
        t = jax.lax.convert_element_type(t, jnp.float32)
        half = self.frequency_embedding_size // 2
        freqs = jnp.exp(
            -jnp.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        args = t[:, None] * freqs[None]
        return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class DiTBlock(nn.Module):
    emb_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        c = nn.gelu(c)
        c = nn.Dense(6 * self.emb_dim, kernel_init=nn.initializers.constant(0.0))(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            c,
            6,
            axis=-1,
        )
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads,
        )(x_mod, x_mod)
        x = x + gate_msa[:, None] * attn_x
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_mod = modulate(x_norm, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(int(self.emb_dim * self.mlp_ratio), self.emb_dim)(x_mod)
        return x + gate_mlp[:, None] * mlp_x


class DiT(nn.Module):
    model_name: str = "DiT"
    emb_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 2.0
    out_dim: int = 256

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        c: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        _, length, _ = x.shape
        pos_emb = self.variable(
            "pos_emb",
            "enc_emb",
            get_1d_sincos_pos_embed,
            self.emb_dim,
            length,
        )
        x = nn.Dense(self.emb_dim)(x) + pos_emb.value
        if c is not None:
            x = x + nn.Dense(self.emb_dim)(c)
        t = TimestepEmbedder(self.emb_dim)(t)
        for _ in range(self.depth):
            x = DiTBlock(self.emb_dim, self.num_heads, self.mlp_ratio)(x, t)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.out_dim)(x)


def decode_at_coords(
    decoder: Decoder,
    decoder_params: Any,
    z: jnp.ndarray,
    coords: jnp.ndarray,
) -> jnp.ndarray:
    def decode_one(coord: jnp.ndarray) -> jnp.ndarray:
        return decoder.apply(decoder_params, z, coord).squeeze(axis=1)

    return jax.vmap(decode_one, in_axes=0, out_axes=1)(coords)

