import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Rotary Positional Embeddings
# -----------------------------
def _get_rope_cache(seq_len: int, head_dim: int, device):
    """Precompute cos/sin for RoPE. head_dim must be even."""
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    half = head_dim // 2
    theta = 10000.0
    inv_freq = torch.pow(theta, -torch.arange(0, half, device=device).float() / half)
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("l,h->lh", t, inv_freq)
    cos = torch.cat([freqs, freqs], dim=-1).cos().unsqueeze(0).unsqueeze(0)
    sin = torch.cat([freqs, freqs], dim=-1).sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    q_rope = (q * cos) + (_rotate_half(q) * sin)
    k_rope = (k * cos) + (_rotate_half(k) * sin)
    return q_rope, k_rope


# -----------------------------
# Multi-Head Self-Attention
# -----------------------------
class MHSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_rope=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        B, L, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, Hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Hd).transpose(1, 2)

        if self.use_rope:
            cos, sin = _get_rope_cache(L, Hd, x.device)
            q, k = apply_rope(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(attn_out)
        out = self.proj_drop(out)
        return (out, attn_weights) if need_weights else (out, None)


# -----------------------------
# Transformer Encoder Block
# -----------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, use_rope=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MHSelfAttention(embed_dim, num_heads, dropout=dropout, use_rope=use_rope)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        key_padding_mask = (mask == 0) if mask is not None else None
        hx = self.ln1(x)
        attn_out, attn_w = self.attn(hx, key_padding_mask=key_padding_mask, need_weights=return_attn)
        x = x + self.dropout(attn_out)
        hx = self.ln2(x)
        ff_out = self.ff(hx)
        x = x + self.dropout(ff_out)
        return x, attn_w


# -----------------------------
# Transformer Encoder
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_layers=8,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        max_len=256,
        num_langs=22,
        phase="phase1",
        use_rope=False,
    ):
        super().__init__()
        self.phase = phase
        self.use_rope = use_rope

        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.use_pos_emb = not use_rope
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(max_len, embed_dim)
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.01)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout, use_rope=use_rope)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(embed_dim)  # 🟢 Added normalization

        if phase == "phase2":
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(embed_dim, num_langs)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if hasattr(self, "token_emb"):
            with torch.no_grad():
                if self.token_emb.padding_idx is not None:
                    self.token_emb.weight[self.token_emb.padding_idx].zero_()

    def set_phase(self, phase, num_langs=None):
        self.phase = phase
        if phase == "phase2" and not hasattr(self, "classifier"):
            if num_langs is None:
                raise ValueError("num_langs must be provided when switching to phase2")
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.token_emb.embedding_dim, num_langs))

    def forward(self, input_ids, mask=None, return_attn=False):
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        if self.use_pos_emb:
            pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
            x = x + self.pos_emb(pos_ids)

        x = self.dropout(x)
        attn_maps = []
        for layer in self.layers:
            x, attn_w = layer(x, mask=mask, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn_w.detach().cpu())

        # Mean pooling with mask
        if mask is not None:
            valid_mask = mask.unsqueeze(-1).type_as(x)
            x_sum = (x * valid_mask).sum(dim=1)
            lengths = valid_mask.sum(dim=1)
            lengths = lengths + (lengths == 0).float()  # avoid divide by zero
            sent_emb = x_sum / lengths
        else:
            sent_emb = x.mean(dim=1)

        sent_emb = self.emb_norm(sent_emb)  # 🟢 Normalization before usage

        if self.phase == "phase1":
            norm = sent_emb.norm(p=2, dim=-1, keepdim=True)
            sent_emb = sent_emb / (norm.clamp(min=1e-8))  # 🟢 Gradient-safe normalization
            return (sent_emb, attn_maps) if return_attn else sent_emb
        else:
            logits = self.classifier(sent_emb)
            return (logits, sent_emb, attn_maps) if return_attn else (logits, sent_emb)


# -----------------------------
# Triplet Loss (Cosine)
# -----------------------------
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=margin
        )

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)
