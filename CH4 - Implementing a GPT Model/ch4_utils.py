import torch
import torch.nn as nn


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        token_emb = self.token_emb(in_idx)

        pos_emb = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )  # This is in the same device as as the input tensor

        x = token_emb + pos_emb
        x = self.drop_emb(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)
        x = self.out_head(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            n_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_short = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_short(x)

        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_short(x)

        x = x + shortcut

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.head_dim = d_out // n_heads
        self.d_out = d_out
        self.context_length = context_length

        self.W_query = torch.nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_out = torch.nn.Linear(self.d_out, self.d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = n_heads
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, context_length, d_in = x.shape
        keys = self.W_key(x)  # (batch, context_length, d_out)
        values = self.W_value(x)
        queries = self.W_query(x)

        keys = keys.view(b, context_length, self.num_heads, self.head_dim)
        values = values.view(b, context_length, self.num_heads, self.head_dim)
        queries = queries.view(b, context_length, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)  # (batch, self.num_heads, context_legnth, head_dim)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attention_score = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:context_length, :context_length]
        attention_score.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_score / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vectors = (attention_weights @ values).transpose(
            1, 2
        )  # (batch, self.num_heads, context_lenght, head_dim) -> (batch, context_lenght, self.num_heads, head_dim)
        context_vectors = context_vectors.contiguous().view(
            b, context_length, self.d_out
        )

        context_vectors = self.W_out(context_vectors)
        return context_vectors


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_counts = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_counts)

        # Retrieve the last token

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
