import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from imports import *


def get_sinusoidal_pos_encoding(length, d_model):
    position = torch.arange(length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def get_continuous_pos_encoding(positions, d_model):
    """Sinusoidal encoding of continuous 2theta positions (not integer indices),
    used for the sparse/peak-list input path."""
    positions = positions / 85.0
    div_term = torch.exp(torch.arange(0, d_model, 2, device=positions.device) * (-math.log(10000.0) / d_model))
    positions = positions.unsqueeze(-1)
    pe = torch.zeros(positions.shape[0], positions.shape[1], d_model, device=positions.device)
    pe[:, :, 0::2] = torch.sin(positions * div_term)
    pe[:, :, 1::2] = torch.cos(positions * div_term)
    return pe


class Attention(nn.Module):
    """
    A multi-head masked attention layer. suitable for both self and cross attention
    """
    def __init__(self, d_model, n_head, attn_pdrop=0.0, resid_pdrop=0.0, att_type='self', linear_bias=False):
        super().__init__()
        assert d_model % n_head == 0
        assert att_type in ['cross', 'self']
        self.att_type = att_type
        # key, query, value projections for all heads
        self.key = nn.Linear(d_model, d_model, bias=linear_bias)
        self.query = nn.Linear(d_model, d_model, bias=linear_bias)
        self.value = nn.Linear(d_model, d_model, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(d_model, d_model, bias=linear_bias)
        self.n_head = n_head

    def forward(self, x, c=None, mask=None, return_attention=False):
        B, N, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)

        query_input = x

        if self.att_type == 'self':
            key_value_input = x
            key_value_N = N
        else:   # self.att_type == 'cross'
            key_value_input = c
            key_value_N = key_value_input.shape[1]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key_value_input).view(B, key_value_N, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, key_value_N, hs)
        q = self.query(query_input).view(B, N, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N, hs)
        v = self.value(key_value_input).view(B, key_value_N, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, key_value_N, hs)
        # causal self-attention; Self-attend: (B, nh, N, hs) x (B, nh, hs, N) -> (B, nh, N, N)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N, key_value_N)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.n_head, N, -1)
            att.masked_fill_(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        if return_attention:
            attention_matrix = att
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, N, key_value_N) x (B, nh, key_value_N, hs) -> (B, nh, N, hs)
        y = y.transpose(1, 2).contiguous().view(B, N, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        # y = y.view(B, N, -1)

        # return
        if return_attention:
            return y, attention_matrix
        else:
            return y


class Tokenizer(nn.Module):
    """
    hierarchical tokenization for XRD data
    """
    def __init__(self, input_dim=8500, window_size=50, stride=25, d_model=32):
        """
        input_dim: input XRD sequence length
        window_size: size of each local window
        stride: stride between windows. If None, uses non-overlapping windows (stride = window_size)
        """
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.stride = stride

        # calculate number of windows
        self.num_windows = (input_dim - window_size) // stride + 1

        #positional encoding
        self.pos_encoding = nn.Parameter(
            get_sinusoidal_pos_encoding(self.num_windows, d_model),
            requires_grad=False  # fixed encoding, not learned
        ) # requires more exploration

        # local processing networks
        self.local_processor = nn.Sequential(
            nn.Linear(window_size, d_model * 2),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model)
        )

        # global context network
        self.global_processor = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.transpose(1, 2)[:, :, 0]
        batch_size = x.size(0)

        # extract overlapping windows
        windows = []
        for i in range(0, x.size(1) - self.window_size + 1, self.stride):
            windows.append(x[:, i:i + self.window_size])
        windows = torch.stack(windows, dim=1)  # [batch_size, num_windows, window_size]

        local_tokens = self.local_processor(windows)  # [batch_size, num_windows, d_model]
        local_tokens = local_tokens + self.pos_encoding.unsqueeze(0)

        global_tokens, _ = self.global_processor(
            local_tokens, local_tokens, local_tokens
        )

        tokens = self.layer_norm(local_tokens + global_tokens)

        return tokens

class BispectrumNoiseSuppress(nn.Module):
    """
    Suppress tiny numerical values in the predicted bispectrum by setting
    them to a small constant instead of zero.
    """
    def __init__(self, threshold=1e-12, min_val=1e-12):
        super().__init__()
        self.threshold = threshold
        self.min_val = min_val

    def forward(self, x):
        # Replace values smaller than threshold with min_val
        mask = torch.abs(x) < self.threshold
        x = torch.where(mask, torch.full_like(x, self.min_val), x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, d_model, h_dim, n_head, attn_pdrop=0.0, resid_pdrop=0.0, att_type='self'):
        super().__init__()
        self.att_type = att_type

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if self.att_type != 'self':
            self.ln_c = nn.LayerNorm(d_model)

        self.attn = Attention(d_model, n_head, attn_pdrop, resid_pdrop, att_type)

        self.mlp = nn.Sequential(nn.Linear(d_model, 2*d_model),
                                 nn.ReLU(True),
                                 nn.Linear(2*d_model, h_dim),
                                 nn.ReLU(True),
                                 nn.Linear(h_dim, d_model),
                                 nn.Dropout(resid_pdrop))

    def forward(self, x_in, c=None, mask=None, return_attention=False):
        if self.att_type != 'self':
            c = self.ln_c(c)

        if return_attention:
            x, attention_matrix = self.attn(self.ln1(x_in), c, mask, return_attention)
            x = x + x_in
        else:
            x = x_in + self.attn(self.ln1(x_in), c, mask)

        x = x + self.mlp(self.ln2(x))

        if return_attention:
            return x, attention_matrix
        else:
            return x
            
class StructuredBispectrumHead(nn.Module):
    def __init__(self, d_model, out_dim, allowed_indices):
        """
        allowed_indices: list of (i, j) tuples indicating valid bispectrum entries
        """
        super().__init__()
        self.out_dim = out_dim
        self.allowed_indices = allowed_indices
        self.n_allowed = len(allowed_indices)

        self.linear = nn.Linear(d_model, self.n_allowed, bias=False)

        # register index tensors (no gradients, moves with device)
        idx = torch.tensor(allowed_indices, dtype=torch.long)
        self.register_buffer("row_idx", idx[:, 0])
        self.register_buffer("col_idx", idx[:, 1])

    def forward(self, x):
        """
        x: (batch, d_model)
        returns: (batch, out_dim[0], out_dim[1])
        """
        batch = x.size(0)

        y = self.linear(x)  # (batch, n_allowed)

        out = x.new_zeros(batch, *self.out_dim)
        out[:, self.row_idx, self.col_idx] = y

        return out


class XRDTransformerEncoder(nn.Module):
    """
    modified transformer with tokenization - supports both dense and sparse XRD input
    """
    def __init__(self, input_dim=8500, out_dim=(10, 35), d_model=32, h_dim=128,
                 n_head=1, n_self_layer=1, n_cross_layer=1, window_size=50,
                 stride=25, ntokens=119, attn_pdrop=0.0, resid_pdrop=0.0,
                 transformer_proc='tokenization', input_type='dense', max_peaks=30,
                 use_intensity=True, allowed_indices=None):
        super().__init__()

        self.transformer_proc = transformer_proc
        self.input_type = input_type  # 'dense' or 'sparse'

        if input_type == 'dense':
            # original dense input processing
            self.tokenizer = Tokenizer(
                input_dim=input_dim,
                window_size=window_size,
                stride=stride,
                d_model=d_model
            )
            self.seq_len = (input_dim - window_size) // stride + 1
            self.lin_xrd = nn.Linear(input_dim, d_model)

        elif input_type == 'sparse':
            # sparse input processing: input is a raw padded peak list
            # [batch_size, max_peaks, 2] of (2theta position, intensity), sorted
            # ascending by position, pad sentinel (-1.0, -1.0). Position (and,
            # if use_intensity, intensity) are embedded here inside the model
            # -- not precomputed in data prep -- so the embedding stays
            # trainable (gradients reach it through the normal backward pass).
            self.seq_len = max_peaks
            self.use_intensity = use_intensity
            if use_intensity:
                self.intensity_embed = nn.Linear(1, d_model)
            self.sparse_attn = nn.MultiheadAttention(d_model, num_heads=n_head, batch_first=True)
            self.sparse_ff = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
        self.self_attn_blocks = nn.Sequential(*[
            TransformerBlock(d_model, h_dim, n_head, attn_pdrop, resid_pdrop, att_type='self')
            for _ in range(n_self_layer)
        ])

        self.cross_attn_blocks = nn.Sequential(*[
            TransformerBlock(d_model, h_dim, n_head, attn_pdrop, resid_pdrop, att_type='cross')
            for _ in range(n_cross_layer)
        ])

        self.pool_cross_block = TransformerBlock(
            d_model, h_dim, n_head, attn_pdrop, resid_pdrop, att_type='cross'
        )

        self.comp_embedding = nn.Embedding(ntokens, d_model)
        self.output_token = nn.Parameter(0.02 * torch.randn(1, 1, d_model))

        self.d_model = d_model
        self.out_dim = out_dim

        self.ln = nn.LayerNorm(d_model)

        # determining whether to use structured head for bispectrum prediction or not
        if allowed_indices is None:
            self.head = nn.Linear(d_model, out_dim[0] * out_dim[1], bias=False)
        else:
            self.head = StructuredBispectrumHead(d_model=d_model, out_dim=out_dim, allowed_indices=allowed_indices)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_pad_c_mask(self, batch, pad_token=0):
        return (batch == pad_token)

    def forward(self, xrd, c=None, padding_mask=None):
        bs = xrd.size(0)

        # Process XRD input based on input type
        if getattr(self, 'input_type', 'dense') == 'dense':
            # Original dense processing
            if self.transformer_proc == 'lin_layer':
                x = self.lin_xrd(xrd)
            elif self.transformer_proc == 'tokenization':
                x = self.tokenizer(xrd)
                
        elif getattr(self, 'input_type', 'dense') == 'sparse':
            # xrd: [batch, max_peaks, 2] raw (position, intensity), pad sentinel -1.0
            positions = xrd[:, :, 0]
            peak_padding_mask = positions < 0  # True = padded slot, ignore

            x = get_continuous_pos_encoding(positions, self.d_model)
            if self.use_intensity:
                intensities = xrd[:, :, 1]
                x = x + self.intensity_embed(intensities.unsqueeze(-1))

            attn_output, *_ = self.sparse_attn(x, x, x, key_padding_mask=peak_padding_mask)
            x = x + attn_output
            ff_output = self.sparse_ff(x)
            x = x + ff_output

            if padding_mask is None:
                padding_mask = peak_padding_mask
        # self-attention
        for self_block in self.self_attn_blocks:
            x = self_block(x, mask=padding_mask)

        if c is not None:
            comp = c.clone().detach()
            c = self.comp_embedding(c[:,:,0].long())

            c_mask = self.create_pad_c_mask(comp[:,:,0].long())

            for cross_block in self.cross_attn_blocks:
                x = cross_block(x, c, mask=c_mask)

        # pool using special output token
        output_token = self.output_token.repeat(bs, 1, 1)
        x_agg = self.pool_cross_block(output_token, x, mask=padding_mask)

        #x_agg = self.linear_out(self.ln(x_agg)).squeeze(1)
        x_agg = self.ln(x_agg).squeeze(1)
        return self.head(x_agg)

        # --- smooth suppression of tiny values ---
        #epsilon = 1e-5  # small threshold in cube-scaled space
        #x_agg = torch.sign(x_agg) * (torch.abs(x_agg) + epsilon) ** 3 / ((torch.abs(x_agg) + epsilon) ** 3 + epsilon ** 3)

        #return x_agg.view(-1, self.out_dim[0], self.out_dim[1])
