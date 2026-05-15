import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import math

from models.Rev_in import RevIN

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class LearnableSWT(nn.Module):
    """
    Learnable 1D Stationary Wavelet Transform.
    Input:  [B, C, L]
    Output: [B, C, level+1, L]  (level details + approximation)
    """
    def __init__(self, in_channels, wavelet_name='db2', level=2, trainable=True):
        super().__init__()
        self.in_channels = in_channels
        self.level = level

        wavelet = pywt.Wavelet(wavelet_name)
        h0 = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
        h1 = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)
        self.kernel_size = h0.shape[-1]

        h0 = h0.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        h1 = h1.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        self.h0 = nn.Parameter(h0, requires_grad=trainable)
        self.h1 = nn.Parameter(h1, requires_grad=trainable)

    def forward(self, x):
        coeffs = []
        approx_coeffs = x
        dilation = 1

        for _ in range(self.level):
            padding = dilation * (self.kernel_size - 1)
            padding_right = padding // 2
            padding_left = padding - padding_right
            approx_coeffs_pad = F.pad(approx_coeffs, (padding_left, padding_right), mode="circular")

            detail_coeff = F.conv1d(
                approx_coeffs_pad, self.h1,
                dilation=dilation, groups=self.in_channels
            )
            approx_coeffs = F.conv1d(
                approx_coeffs_pad, self.h0,
                dilation=dilation, groups=self.in_channels
            )

            coeffs.append(detail_coeff)
            dilation *= 2

        coeffs.append(approx_coeffs)
        return torch.stack(list(reversed(coeffs)), dim=2)


class LearnableISWT(nn.Module):
    """
    Learnable 1D Inverse Stationary Wavelet Transform.
    Input:  [B, C, level+1, L]
    Output: [B, C, L]
    """
    def __init__(self, in_channels, wavelet_name='db2', level=2, trainable=True):
        super().__init__()
        self.in_channels = in_channels
        self.level = level

        wavelet = pywt.Wavelet(wavelet_name)
        g0 = torch.tensor(wavelet.rec_lo[::-1], dtype=torch.float32)
        g1 = torch.tensor(wavelet.rec_hi[::-1], dtype=torch.float32)
        self.kernel_size = g0.shape[-1]

        g0 = g0.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        g1 = g1.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        self.g0 = nn.Parameter(g0, requires_grad=trainable)
        self.g1 = nn.Parameter(g1, requires_grad=trainable)

    def forward(self, coeffs):
        approx_coeff = coeffs[:, :, 0, :]
        detail_coeffs = coeffs[:, :, 1:, :]

        dilation = 2 ** (self.level - 1)

        for i in range(self.level):
            detail_coeff = detail_coeffs[:, :, i, :]

            padding = dilation * (self.kernel_size - 1)
            padding_left = (dilation * self.kernel_size) // 2
            pad = (padding_left, padding - padding_left)

            approx_coeff_pad = F.pad(approx_coeff, pad, mode="circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, mode="circular")

            y_approx = F.conv1d(
                approx_coeff_pad, self.g0,
                groups=self.in_channels, dilation=dilation
            )
            y_detail = F.conv1d(
                detail_coeff_pad, self.g1,
                groups=self.in_channels, dilation=dilation
            )

            approx_coeff = (y_approx + y_detail) / 2.0
            dilation //= 2

        return approx_coeff


class Patching(nn.Module):
    """
    Non-overlapping patching along time.
    Input:  [B, C, L]
    Output: patches [B, C, N, P], pad_len
    """
    def __init__(self, patch_len):
        super().__init__()
        self.patch_len = patch_len

    def forward(self, x):
        seq_len = x.shape[-1]
        pad_len = (self.patch_len - (seq_len % self.patch_len)) % self.patch_len
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        num_patches = x.shape[-1] // self.patch_len
        x_patched = x.view(*x.shape[:-1], num_patches, self.patch_len)
        return x_patched, pad_len



class HierarchicalMixerBlock(nn.Module):
    """
    A Hierarchical Inter-Patch Mixer block that performs temporal mixing.
    Uses a single relational branch to learn dependencies between adjacent blocks.
    """
    def __init__(self, patch_len, d_model, dropout_val, max_mix_levels):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.dropout_val = dropout_val
        self.max_mix_levels = max_mix_levels 

        self.level_mlps = nn.ModuleList() 
        
        self.distribution_weights = nn.ParameterList() 
        self.sigmoid = nn.Sigmoid()

        for k in range(1, max_mix_levels + 1):
            block_size = (2**(k - 1)) * patch_len 
            
            input_size = 2 * block_size
            
        
            mlp_inter = nn.Sequential(
                nn.Linear(input_size, d_model),
                nn.GELU(),
                nn.Dropout(p=dropout_val),
                nn.Linear(d_model, block_size)
            )
            self.level_mlps.append(mlp_inter)
            
            self.distribution_weights.append(nn.Parameter(torch.zeros(1)))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, C, N, P = patches.shape
        mixed_patches = patches
        
        if N < 2:
            return patches
        
        max_k = int(torch.floor(torch.log2(torch.tensor(N).float())))
        max_k = min(max_k, self.max_mix_levels) 

        for k in range(1, max_k + 1):
            agg_factor = 2**(k - 1)
            block_size = agg_factor * P 
            N_blocks = N // agg_factor 
            
            if N_blocks < 2:
                break
                
            N_pairs = N_blocks - 1 
            N_mixable_blocks = N_pairs + 1 
            N_mixable_small_patches = N_mixable_blocks * agg_factor
            
            M = mixed_patches[:, :, :N_mixable_small_patches, :]
            U = mixed_patches[:, :, N_mixable_small_patches:, :]
            
            Q = M.reshape(B, C, N_mixable_blocks, block_size)
            
            Q_left = Q[:, :, :-1, :]       
            Q_right = Q[:, :, 1:, :]       
            Q_pairs = torch.cat([Q_left, Q_right], dim=-1) 
            
            hier_rel = self.level_mlps[k-1](Q_pairs)
            
            alpha = self.sigmoid(self.distribution_weights[k-1])
            
            Q_left_update = Q_left + alpha * hier_rel
            Q_right_update = Q_right + (1.0 - alpha) * hier_rel
            
            Q_updated = Q.clone()
            Q_updated[:, :, :-1, :] = Q_left_update  
            Q_updated[:, :, 1:, :] = Q_right_update 
            
            Q = Q_updated 
            
            Q_base_patches = Q.reshape(B, C, N_mixable_small_patches, P)
            
            if U.numel() > 0:
                mixed_patches = torch.cat([Q_base_patches, U], dim=2)
            else:
                mixed_patches = Q_base_patches
                
        return mixed_patches

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_revin = configs.revin
        self.dropout_val = configs.dropout
        self.patch_len = configs.patch_len
        self.num_levels = configs.wavelet_j
        self.num_scales = configs.wavelet_j + 1
        self.wavelet_name = configs.wavelet
        self.d_ff = configs.d_ff
        
        self.max_mix_levels = configs.e_layers

        self.rev_norm = RevIN(self.enc_in, affine=configs.affine)

        self.swt_decomp = LearnableSWT(
            in_channels=self.enc_in,
            wavelet_name=self.wavelet_name,
            level=self.num_levels,
            trainable=True
        )
        self.swt_recon = LearnableISWT(
            in_channels=self.enc_in,
            wavelet_name=self.wavelet_name,
            level=self.num_levels,
            trainable=True
        )

        self.patching = Patching(self.patch_len)

        N_patches = math.ceil(self.seq_len / self.patch_len)
        self.L_patched = int(N_patches * self.patch_len)
        
        self.enc_embeddings = nn.ModuleList()
        self.encoders = nn.ModuleList() 
        self.encoder_out_projs = nn.ModuleList() 

        for _ in range(self.num_levels):
            self.enc_embeddings.append(
                DataEmbedding_inverted( 
                    self.seq_len,
                    configs.d_model,
                    configs.embed,
                    configs.freq,
                    configs.fc_dropout
                )
            )
            self.encoders.append(
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                FullAttention(
                                    False,
                                    configs.factor,
                                    attention_dropout=self.dropout_val,
                                    output_attention=configs.output_attention
                                ),
                                self.d_model,
                                configs.n_heads 
                            ),
                            self.d_model,
                            self.d_ff,
                            dropout=self.dropout_val,
                            activation=configs.activation
                        ) for l in range(configs.e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(self.d_model)
                )
            )
            self.encoder_out_projs.append(
                nn.Linear(self.d_model, self.seq_len)
            )


        self.intra_patch_mlps = nn.ModuleList()
        self.mixer_blocks = nn.ModuleList() 
        self.batch_norms = nn.ModuleList() 
        self.projection_layers = nn.ModuleList() 

        for _ in range(self.num_scales):
            self.intra_patch_mlps.append(
                nn.Sequential(
                    nn.Linear(self.patch_len, self.d_model),
                    nn.GELU(),
                    nn.Dropout(p=self.dropout_val),
                    nn.Linear(self.d_model, self.patch_len)
                )
            )

            self.mixer_blocks.append(
                HierarchicalMixerBlock(
                    self.patch_len, 
                    self.d_model, 
                    self.dropout_val, 
                    self.max_mix_levels
                )
            )

            self.batch_norms.append(nn.BatchNorm1d(self.enc_in))

            self.projection_layers.append(
                nn.Linear(self.L_patched, self.seq_len)
            )

        self.mlp_residual = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.GELU(),
            nn.Dropout(p=self.dropout_val),
            nn.Linear(self.d_model, self.seq_len)
        )

        self.pred_layer = nn.Linear(self.seq_len, self.pred_len)

    def _process_scale(self, scale_coeffs, m):
        """
        Process one SWT scale (approx or detail) with the unified mixing block.
        scale_coeffs: [B, C, L]
        """
        B, C, L = scale_coeffs.shape

        patches, pad_len = self.patching(scale_coeffs)
        Bp, Cp, N, P = patches.shape

        residual = patches
        patches_reshaped = patches.reshape(Bp * Cp * N, P)
        patches_reshaped = self.intra_patch_mlps[m](patches_reshaped)
        patches = patches_reshaped.reshape(Bp, Cp, N, P) + residual
        
        patches = self.mixer_blocks[m](patches)

        patches_flat = patches.reshape(Bp, Cp, -1) 
        
        processed_scale = self.projection_layers[m](patches_flat)
        
        return processed_scale

    def forward(self, x, x_mark_enc=None):
        """
        x: [B, L, C]
        """
        B, L, C = x.shape
        assert L == self.seq_len, "Input length must equal configs.seq_len"

        if self.use_revin:
            x_norm = self.rev_norm(x, 'norm')
        else:
            x_norm = x

        x_norm = x_norm.permute(0, 2, 1)

        coeffs = self.swt_decomp(x_norm)

        processed_coeffs_list = []
        for m in range(self.num_scales):
            scale_coeffs = coeffs[:, :, m, :]

            if m < self.num_levels: 
                
                
                coeffs_for_enc = scale_coeffs.permute(0, 2, 1)
                
                enc_input = self.enc_embeddings[m](coeffs_for_enc, x_mark=None)

                enc_output, _ = self.encoders[m](enc_input, attn_mask=None)
                
                encoder_out = self.encoder_out_projs[m](enc_output)
                scale_coeffs = scale_coeffs + encoder_out

            scale_coeffs = self.batch_norms[m](scale_coeffs) 

            processed_scale = self._process_scale(scale_coeffs, m)
            processed_coeffs_list.append(processed_scale)

        processed_coeffs = torch.stack(processed_coeffs_list, dim=2)

        reconstructed = self.swt_recon(processed_coeffs)

        mlp_out = self.mlp_residual(reconstructed)
        out_residual = reconstructed + mlp_out

        forecast = self.pred_layer(out_residual)

        out = forecast.permute(0, 2, 1)

        if self.use_revin:
            out = self.rev_norm(out, 'denorm')

        return out
