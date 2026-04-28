import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len, base=10000):
        super(RoPE, self).__init__()
        theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, theta)
        self.register_buffer('cos', freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin', freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        T, D = x.shape[-2], x.shape[-1]
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        cos = self.cos[:, :, :T].to(x.dtype)
        sin = self.sin[:, :, :T].to(x.dtype)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        self.dropout = dropout

        self.W_qkv = nn.Linear(self.d_model, 3*self.d_model, bias=False)
        self.out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rope = RoPE(self.d_head, max_seq_len)

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = self.W_qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q = self.rope(Q)
        K = self.rope(K)
        attn = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(attn)


class SwiGLU(nn.Module):
    def __init__(self, d_model, ff_dim):
        super(SwiGLU, self).__init__()
        self.W_gate = nn.Linear(d_model, ff_dim, bias=False)
        self.W_up = nn.Linear(d_model, ff_dim, bias=False)
        self.W_down = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, 8 * d_model // 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


if __name__ == "__main__":
    config = dict(
        vocab_size=50304,
        d_model=768,
        n_heads=12,
        n_layers=12,
        max_seq_len=2048,
        dropout=0.1,
    )
    model = Transformer(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params / 1e6:.2f}M")

    B, T = 2, 128
    idx = torch.randint(0, config["vocab_size"], (B, T))
    logits = model(idx)
    print(f"Input:  {tuple(idx.shape)}")
    print(f"Output: {tuple(logits.shape)}")
