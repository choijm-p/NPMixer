import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
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
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

from math import sqrt

# Original Diff Transfomrers

class FullAttention_full(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=None, n_heads=None, layer_index=2):
        super(FullAttention_full, self).__init__()
        self.scale = scale or 1.0
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.layer_index = layer_index
        self.n_heads = n_heads
        self.d_model = d_model

        D = self.d_model // self.n_heads  # Dimension per head
        self.head_dim = D

        # Initialize lambda_init
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (self.layer_index - 1))
        

        # RMSNorm over 2 * head_dim
        self.norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # Learnable parameters for lambda computation (per head and per dimension)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.n_heads, self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.n_heads, self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.n_heads, self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.n_heads, self.head_dim).normal_(mean=0, std=0.1))

    def forward(self, queries, keys, values, attn_mask):
        # queries, keys, values: (B, L/S, H, 2, D)
        B, L, H, _, D = queries.shape
        _, S, _, _, _ = keys.shape

        scale = self.scale / sqrt(D)

        # Compute attention scores
        attn_scores = torch.einsum('blhid,bshid->bhils', queries * scale, keys)  # (B, H, 2, L, S)

        if self.mask_flag and attn_mask is not None:
            attn_scores += attn_mask.unsqueeze(1).unsqueeze(2)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, 2, L, S)

        # Separate the two components
        attn_weights1 = attn_weights[:, :, 0, :, :]  # (B, H, L, S)
        attn_weights2 = attn_weights[:, :, 1, :, :]  # (B, H, L, S)

        # Compute lambda_1 and lambda_2 per head
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))  # Shape: (H,)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))  # Shape: (H,)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init  # Shape: (H,)

        # Reshape lambda_full for broadcasting
        lambda_full = lambda_full.view(1, H, 1, 1)  # Shape: (1, H, 1, 1)

        # Combine attention weights using learnable lambda
        attn_weights_combined = attn_weights1 - lambda_full * attn_weights2
        attn_weights_combined = self.dropout(attn_weights_combined)

        # Prepare values
        values = values.permute(0, 2, 3, 1, 4).contiguous()  # (B, H, 2, S, D)
        values = values.view(B, H, 2, S, D)
        values = values.permute(0, 1, 3, 2, 4).contiguous().view(B, H, S, 2 * D)  # (B, H, S, 2D)

        # Compute attention output
        attn_output = torch.matmul(attn_weights_combined, values)  # (B, H, L, 2D)

        # Apply RMSNorm over last dimension
        attn_output = attn_output.view(-1, 2 * D)
        attn_output = self.norm(attn_output)
        attn_output = attn_output.view(B, H, L, 2 * D)

        # Scale attention output
        attn_output = attn_output * (1 - self.lambda_init)

        # Reshape to [B, L, H, 2D]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, L, H, 2D)

        if self.output_attention:
            return (attn_output, attn_weights_combined)
        else:
            return (attn_output, None)


class AttentionLayer_full(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer_full, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_keys = d_keys
        self.d_values = d_values

        self.inner_attention = attention

        # Projections now output features for n_heads * 2
        self.query_projection = nn.Linear(d_model, d_keys * n_heads * 2)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads * 2)
        self.value_projection = nn.Linear(d_model, d_values * n_heads * 2)

        self.out_projection = nn.Linear(n_heads * 2 * d_values, d_model)  # Adjusted to map from H * 2D to d_model

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        D = self.d_keys  # Dimension per head

        # Project and reshape queries, keys, values
        queries = self.query_projection(queries).view(B, L, H, 2, D)
        keys = self.key_projection(keys).view(B, S, H, 2, D)
        values = self.value_projection(values).view(B, S, H, 2, D)

        # Transpose dimensions for attention computation
        queries = queries.permute(0, 1, 2, 3, 4)  # (B, L, H, 2, D)
        keys = keys.permute(0, 1, 2, 3, 4)        # (B, S, H, 2, D)
        values = values.permute(0, 1, 2, 3, 4)    # (B, S, H, 2, D)

        # Apply attention
        out, attn = self.inner_attention(queries, keys, values, attn_mask)

        # Reshape and project output
        # out: [B, L, H, 2D]
        out = out.contiguous().view(B, L, H * 2 * D)  # [B, L, H * 2D]
        out = self.out_projection(out)  # [B, L, d_model]

        return out, attn

class FullAttentionN(nn.Module):
    """
    Generalized 'FullAttention' that handles n separate softmax branches.

    Formula for combined attention:
      attn_weights_combined = softmax_1
                              - lambda_1 * softmax_2
                              - lambda_2 * softmax_3
                              ...
                              - lambda_(n-1) * softmax_n
    """
    def __init__(
        self,
        n_components=3,             # Number of "softmax branches" (was 2 in the original)
        mask_flag=True,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        d_model=None,
        n_heads=None,
        layer_index=2
    ):
        """
        Args:
            n_components: number of softmax branches (n). If n=2, this is the original logic.
        """
        super(FullAttentionN, self).__init__()
        self.n_components = n_components
        self.scale = scale or 1.0
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.layer_index = layer_index
        self.n_heads = n_heads
        self.d_model = d_model

        # Dimension per head
        D = self.d_model // self.n_heads
        self.head_dim = D

        # A base "lambda_init" as in the original code
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (self.layer_index - 1))

        # RMSNorm over n * head_dim (because we have n components of dimension D per head)
        self.norm = RMSNorm(self.n_components * self.head_dim, eps=1e-5, elementwise_affine=True)

        # --------------------------------------------------
        # We want (n-1) lambdas: each lambda_i is derived from
        #    lambda_q[i] and lambda_k[i] via:
        #    lambda_i = exp( sum( lambda_q[i] * lambda_k[i], dim=-1 ) )
        #
        # So we store them in ParameterLists, each shape: (n_heads, head_dim).
        # We'll index them from 0 .. (n-2) to correspond to the 2nd..n-th branch.
        # --------------------------------------------------
        self.lambda_q_list = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_heads, self.head_dim).normal_(mean=0, std=0.1))
            for _ in range(self.n_components - 1)
        ])
        self.lambda_k_list = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_heads, self.head_dim).normal_(mean=0, std=0.1))
            for _ in range(self.n_components - 1)
        ])

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries, keys, values shape: (B, L, H, n, D)
            B = batch size
            L = query length
            S = key/value length
            H = number of heads
            n = self.n_components
            D = self.head_dim
        """
        B, L, H, n, D = queries.shape
        _, S, _, _, _ = keys.shape

        # Scale factor
        scale = self.scale / sqrt(D)

        # ---- Compute attention scores ----
        # attn_scores: (B, H, n, L, S)
        # We sum over the dimension 'D' in queries and keys, so we do
        # query * key -> shape (B, H, n, L, S)
        # Using Einstein summation:
        #   queries:  (B, L, H, n, D)
        #   keys:     (B, S, H, n, D)
        # We want to match the B, H, n dimensions, multiply over D, and keep L, S.
        attn_scores = torch.einsum('blhnd,bshnd->bhnls', queries * scale, keys)

        # ---- Optional mask ----
        if self.mask_flag and (attn_mask is not None):
            # attn_mask shape might be (B, L, S) or (L, S)
            # so we broadcast to (B, 1, n, L, S). This ensures adding it to 'attn_scores'.
            attn_scores += attn_mask.unsqueeze(1).unsqueeze(2)

        # ---- Softmax along the last dim (S) but keep the n dimension separate ----
        # attn_weights shape: (B, H, n, L, S)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        


        # ---- Split each of the n softmax branches ----
        # attn_weights[i] -> (B, H, L, S) for i in range(n)
        # We'll combine them: w1 - λ1*w2 - λ2*w3 - ... - λ(n-1)*w(n)
        # We'll start with w1, then subtract λ1 * w2, etc.
        w_first = attn_weights[:, :, 0, :, :]  # shape: (B, H, L, S)
        attn_weights_combined = w_first.clone()

        # For i in [1..n-1], compute lambda_i and subtract from the combination
        for i in range(1, self.n_components):
            w_i = attn_weights[:, :, i, :, :]  # shape: (B, H, L, S)

            # Compute lambda_i for this branch i (where i-1 is the index in the ParameterList)
            #   lambda_i = exp( sum_{d}( lambda_q[i-1]*lambda_k[i-1] ) )
            # shape of lambda_i across heads: (H,)
            lambda_q = self.lambda_q_list[i - 1]  # shape (H, D)
            lambda_k = self.lambda_k_list[i - 1]  # shape (H, D)
            lambda_i = torch.exp(torch.sum(lambda_q * lambda_k, dim=-1))  # (H,)

            # Reshape for broadcast: (1, H, 1, 1)
            lambda_i = lambda_i.view(1, H, 1, 1)

            # Subtract from the combined weights
            attn_weights_combined = attn_weights_combined - lambda_i * w_i

        # ---- Dropout on combined weights ----
        attn_weights_combined = self.dropout(attn_weights_combined)

        # ---- Prepare values for multiplication ----
        # Original shape of values: (B, S, H, n, D)
        # We want to flatten the n and D dimensions for each head: shape -> (B, H, S, n*D)
        values = values.permute(0, 2, 3, 1, 4).contiguous()  # (B, H, n, S, D)
        values = values.view(B, H, n, S, D)
        values = values.permute(0, 1, 3, 2, 4).contiguous().view(B, H, S, n * D)

        # ---- Apply the combined attention weights to the values ----
        # attn_output: (B, H, L, n*D)
        attn_output = torch.matmul(attn_weights_combined, values)

        # ---- Apply RMSNorm over the last dimension (n*D) ----
        attn_output = attn_output.view(-1, self.n_components * self.head_dim)
        attn_output = self.norm(attn_output)
        attn_output = attn_output.view(B, H, L, self.n_components * self.head_dim)

        # ---- Scale final output by (1 - lambda_init), as in the original code ----
        attn_output = attn_output * (1 - self.lambda_init)

        # ---- Reshape to (B, L, H, n*D) so that subsequent ops can proceed normally ----
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, L, H, n*D)

        # Return attention if requested
        if self.output_attention:
            return attn_output, attn_weights_combined
        else:
            return attn_output, None


class AttentionLayerN(nn.Module):
    """
    Corresponding Attention Layer that projects queries, keys, values into
    (n_components) sub-channels, then calls FullAttentionN.
    """
    def __init__(self, attention, d_model, n_heads, n_components=3, d_keys=None, d_values=None):
        """
        Args:
            attention: an instance of FullAttentionN (or similar) 
            d_model: total model dimensionality
            n_heads: number of attention heads
            n_components: how many "softmax branches" 
        """
        super(AttentionLayerN, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_keys = d_keys
        self.d_values = d_values
        self.n_components = n_components

        self.inner_attention = attention  # e.g. FullAttentionN(...) from above

        # Each projection now outputs n_components * (head_dim)
        # because for each head, we have n_components sub-channels
        self.query_projection = nn.Linear(d_model, d_keys * n_heads * n_components)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads * n_components)
        self.value_projection = nn.Linear(d_model, d_values * n_heads * n_components)

        # The output projection is from (n_heads * n_components * d_values) back to d_model
        self.out_projection = nn.Linear(n_heads * n_components * d_values, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries, keys, values: (B, L, d_model), (B, S, d_model), (B, S, d_model)
        attn_mask: optional mask of shape (B, L, S) or (L, S)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        Dk = self.d_keys   # dimension per head for Q/K
        Dv = self.d_values # dimension per head for V
        nC = self.n_components

        # ---- Project Q, K, V ----
        # shape after linear: (B, L, H*nC*Dk) for queries
        proj_queries = self.query_projection(queries)  # (B, L, H*nC*Dk)
        proj_keys = self.key_projection(keys)          # (B, S, H*nC*Dk)
        proj_values = self.value_projection(values)    # (B, S, H*nC*Dv)

        # ---- Reshape to (B, L, H, nC, Dk) for queries, similarly for K/V ----
        proj_queries = proj_queries.view(B, L, H, nC, Dk)  # (B, L, H, nC, Dk)
        proj_keys = proj_keys.view(B, S, H, nC, Dk)        # (B, S, H, nC, Dk)
        proj_values = proj_values.view(B, S, H, nC, Dv)    # (B, S, H, nC, Dv)

        # ---- Transpose/permute if needed by FullAttentionN ----
        # We'll keep them in (B, L, H, nC, Dk) so it matches the einsum in FullAttentionN
        # and pass them directly. The FullAttentionN expects (B, L, H, n, D).
        out, attn = self.inner_attention(proj_queries, proj_keys, proj_values, attn_mask)
        # out is (B, L, H, nC*Dv)

        # ---- Final linear projection to get back to d_model ----
        out = out.contiguous().view(B, L, H * nC * Dv)  # flatten heads & components
        out = self.out_projection(out)  # -> (B, L, d_model)

        return out, attn

class ChannelMixing(nn.Module):
    """
    Channel-Mixing attention over the (B, L, C) input.
    By default, we assume attention_dim == number of channels
    to keep a direct residual connection. If not, you can
    project back to the original dimension.
    """
    def __init__(self, channels, attention_dim=None):
        super(ChannelMixing, self).__init__()
        self.channels = channels
        self.attention_dim = attention_dim or channels

        # Q, K, V linear projections
        self.W_q = nn.Linear(channels, self.attention_dim, bias=False)
        self.W_k = nn.Linear(channels, self.attention_dim, bias=False)
        self.W_v = nn.Linear(channels, self.attention_dim, bias=False)

        # If attention_dim != channels, add a final proj back to channels
        if self.attention_dim != self.channels:
            self.proj_back = nn.Linear(self.attention_dim, self.channels, bias=False)
        else:
            self.proj_back = None

    def forward(self, x):
        """
        x: Tensor of shape [B, L, C]
        Returns:
            out: Tensor of shape [B, L, C], after attention over channels.
        """
        B, L, C = x.shape

        # Project into Q, K, V
        # Each shape: [B, L, attention_dim]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Compute attention scores: Q * K^T
        # => shape [B, L, L] if we treat L as the 'sequence' dimension
        # BUT if you truly want "channel mixing," you might interpret C as the dimension to be attended over.
        # The example below stays consistent with the shape [B, L, attention_dim] and attends along L.
        # 
        # If you want to attend along the 'channels' dimension, you'd do a transpose or reshape. 
        # For demonstration, let's keep it along L, as in typical self-attention:
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, L, L]
        scores = scores / math.sqrt(self.attention_dim)

        # Softmax over the last dimension (the "L" dimension)
        attn = F.softmax(scores, dim=-1)  # [B, L, L]

        # Weighted sum
        out = torch.matmul(attn, V)  # [B, L, attention_dim]

        # If attention_dim != channels, project back
        if self.proj_back is not None:
            out = self.proj_back(out)  # map back to [B, L, C]

        # Residual connection
        out = out + x

        return out
    
##########################
# Channel Attention      #
##########################

class FullAttentionChannel(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttentionChannel, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries, keys, values: [B, C, L, H, E]
        Returns:
            V: [B, C, L, H, E]
        """
        B, C, L, H, E = queries.shape
        _, _, S, _, _ = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        
        # Compute attention scores, preserving channel dimension.
        scores = torch.einsum("bclhe,bcshe->bchls", queries, keys)
        
        if self.mask_flag:
            if attn_mask is None:
                # Here you could define a mask that works with channels.
                attn_mask = torch.zeros(B, C, L, S, device=queries.device).bool()
            scores.masked_fill_(attn_mask, -float('inf'))
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bchls,bcshe->bclhe", A, values)
        
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None