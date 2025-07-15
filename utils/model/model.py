# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/


import torch
import torch.nn as nn
import torch.nn.functional as F

def load_model(model_path, config, device):
    
    device = torch.device(device)
    
    # Retrieve the half precision setting from the config
    use_half_precision = config.get('use_half_precision', True)
    
    # 🔥 NEW: Check for CUDA and cuDNN availability.
    # If half precision is requested but CUDA or cuDNN are not available,
    # fall back to full precision and update the config.
    if use_half_precision:
        if not (device.type == 'cuda' and torch.cuda.is_available() and torch.backends.cudnn.enabled):
            print("⚠ Half-precision requested but CUDA or cuDNN not available. Falling back to full precision.")
            use_half_precision = False
            config['use_half_precision'] = False  # Update config to reflect the fallback

    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    num_heads = config['num_heads']
    
    encoder = Encoder(config['input_dim'], hidden_dim, n_layers, num_heads)
    decoder = Decoder(config['output_dim'], hidden_dim, n_layers, num_heads)
    model = Seq2Seq(encoder, decoder, device).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # Convert the model to half precision if applicable
    if use_half_precision and device.type == 'cuda':
        model = model.to(torch.float16)
        print("⚡ Model converted to float16 (half-precision).")
    else:
        print("🚫 Half-precision not applied (CPU or unsupported GPU or False set in config).")

    model.eval()
    return model



# -------------------------------------------------------------------------------------------
# Seq2Seq Model
# -------------------------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src):
        encoder_outputs = self.encoder(src)
        output = self.decoder(encoder_outputs)
        return output

# -------------------------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE) for Local Attention
# -------------------------------------------------------------------------------------------
def apply_rope_qk(q, k, use_local_positional_encoding=True):
    if not use_local_positional_encoding:
        return q, k  # Return unmodified q, k if RoPE is disabled

    batch_size, num_heads, seq_len, head_dim = q.size()
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    position = torch.arange(seq_len, dtype=torch.float, device=q.device).unsqueeze(1)  # (seq_len, 1)
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float, device=q.device)  # (head_dim // 2)
    div_term = torch.exp(-torch.log(torch.tensor(10000.0)) * dim_indices / head_dim)

    angle = position * div_term  # (seq_len, head_dim // 2)
    sin = torch.sin(angle).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
    cos = torch.cos(angle).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)

    def rope_transform(x):
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split into even and odd parts
        x_rope_even = x1 * cos - x2 * sin
        x_rope_odd = x1 * sin + x2 * cos
        return torch.stack([x_rope_even, x_rope_odd], dim=-1).flatten(-2)

    q = rope_transform(q)
    k = rope_transform(k)
    return q, k


# -------------------------------------------------------------------------------------------
# Multi-Head Attention with RoPE
# -------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention requires PyTorch >= 2.0")

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        # Reshape to (B, H, L, D)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (if enabled)
        query, key = apply_rope_qk(query, key)

        if self.flash:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
            attn_weights = None
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_linear(attn_output)
        output = self.resid_dropout(output)

        return output, attn_weights

# -------------------------------------------------------------------------------------------
# Feed-Forward Network
# -------------------------------------------------------------------------------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim, dim_feedforward=2048, dropout=0.0):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# -------------------------------------------------------------------------------------------
# Custom Transformer Encoder/Decoder
# -------------------------------------------------------------------------------------------
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, 4 * hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2, _ = self.self_attn(src, src, src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, 4 * hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _ = self.multihead_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# -------------------------------------------------------------------------------------------
# Encoder 
# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, num_heads, dropout=0.0, use_norm=True):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # CHANGED: Removed global positional encoding as RoPE is used in MHA.
        self.transformer_encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(hidden_dim, num_heads, dropout) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_norm else None

    def forward(self, x):
        x = self.embedding(x)
        # CHANGED: Global positional encoding removed.
        for layer in self.transformer_encoder:
            x = layer(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        return x

# -------------------------------------------------------------------------------------------
# Decoder 
# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, num_heads, dropout=0.0, use_norm=True):
        super(Decoder, self).__init__()
        self.transformer_decoder = nn.ModuleList([
            CustomTransformerDecoderLayer(hidden_dim, num_heads, dropout) for _ in range(n_layers)
        ])
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_norm else None

    def forward(self, encoder_outputs):
        x = encoder_outputs 
        for layer in self.transformer_decoder:
            x = layer(x, encoder_outputs)
        if self.layer_norm:
            x = self.layer_norm(x)
        return self.fc_output(x)


