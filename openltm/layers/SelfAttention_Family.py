import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import repeat
from layers.Attn_Bias import BinaryAttentionBias
from layers.Attn_Projection import QueryKeyProjection, RotaryProjection
from utils.masking import TriangularCausalMask, TimerMultivariateMask, TimerCovariateMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


def path_scores(q, k, w, beta, n_vars, n_tokens):
    """PaTH scores q_{c,i}^T H_i ... H_{j+1} k_{c',j}, H_t = I - beta_t w_t w_t^T taken from the key variable's own token stream.
    q, k, w (B H C*N E) in Timer-XL's variable-major token order, |w| = 1; beta (B H C*N).  Returns (B H C*N C*N); entries with
    key time > query time are 0 (masked by the caller)."""
    B, H, L, E = q.shape
    C, N = n_vars, n_tokens
    q, k, w = (t.reshape(B, H, C, N, E) for t in (q, k, w))
    beta = beta.reshape(B, H, C, N)
    S = q.new_zeros(B, H, C, N, C, N)
    Kt = k[:, :, :, :1]                                                    # Kt[..., j, :] = H_i ... H_{j+1} k_j
    for i in range(N):
        if i > 0:
            wi = w[:, :, :, i:i + 1]
            Kt = Kt - beta[:, :, :, i, None, None] * wi * (Kt * wi).sum(-1, keepdim=True)
            Kt = torch.cat([Kt, k[:, :, :, i:i + 1]], 3)
        S[:, :, :, i, :, :i + 1] = torch.einsum('bhce,bhdje->bhcdj', q[:, :, :, i], Kt)
    return S.reshape(B, H, L, L)


class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, covariate=False, flash_attention=False, pos='rope'):
        super(TimeAttention, self).__init__()
        assert pos in ('rope', 'nope', 'path', 'rope_path')
        self.pos = pos
        self.needs_x = 'path' in pos
        if self.needs_x:                                                   # PaTH: w_t, beta_t from the key token's layer input
            assert not flash_attention
            self.num_heads = num_heads
            self.w_proj = nn.Linear(d_model, d_model)
            self.beta_proj = nn.Linear(d_model, num_heads)
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None, x=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        if 'rope' in self.pos:
            queries, keys = self.qk_proj(
                queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            if self.needs_x:
                w = nn.functional.normalize(self.w_proj(x).view(B, S, H, E), dim=-1).permute(0, 2, 1, 3)
                beta = 2 * torch.sigmoid(self.beta_proj(x)).permute(0, 2, 1)
                scores = path_scores(queries, keys, w, beta, n_vars, n_tokens)
            else:
                scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        extra = {'x': keys} if getattr(self.inner_attention, 'needs_x', False) else {}
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens,
            tau=tau,
            delta=delta,
            **extra
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

