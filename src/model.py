import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_causal_mask(sz, device=None):
    mask = torch.tril(torch.ones((sz, sz), dtype=torch.bool, device=device))
    return mask  # True 表示可见

def create_padding_mask(seq, pad_token=0):
    # seq: (B, T) -> (B,1,1,T)
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask_bool = mask != 0
            else:
                mask_bool = mask
            scores = scores.masked_fill(~mask_bool, float('-1e9'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        return x

    def _combine_heads(self, x):
        x = x.transpose(1, 2).contiguous()
        B, T, _, _ = x.size()
        return x.view(B, T, self.d_model)

    def forward(self, q, k, v, mask=None):
        Q = self._split_heads(self.w_q(q))
        K = self._split_heads(self.w_k(k))
        V = self._split_heads(self.w_v(v))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.expand(Q.size(0), self.num_heads, -1, -1)
        out, attn = self.attention(Q, K, V, mask)
        out = self._combine_heads(out)
        out = self.fc(out)
        out = self.dropout(out)
        return out, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn = self.mha(x, x, x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x, attn

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.token_emb(x) * math.sqrt(self.token_emb.embedding_dim)
        x = self.pos_enc(x)
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attentions.append(attn)
        x = self.norm(x)
        return x, attentions

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        sa_out, sa = self.self_attn(x, x, x, mask=tgt_mask)
        x = x + self.dropout(sa_out)
        x = self.norm1(x)

        mem_mask = memory_mask
        if memory_mask is not None:
            # 确保 memory_mask 是 (B,1,1,T_src)
            if memory_mask.dim() == 2:
                mem_mask = memory_mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,T_src)
            elif memory_mask.dim() == 3:
                mem_mask = memory_mask.unsqueeze(1)  # (B,1,T_tgt,T_src)
            elif memory_mask.dim() == 4:
                mem_mask = memory_mask
            mem_mask = mem_mask.expand(-1, 1, x.size(1), -1)  # B,1,T_tgt,T_src
        ca_out, ca = self.enc_dec_attn(x, enc_out, enc_out, mask=mem_mask)
        x = x + self.dropout(ca_out)
        x = self.norm2(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)
        return x, sa, ca

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        x = self.token_emb(tgt_ids) * math.sqrt(self.token_emb.embedding_dim)
        x = self.pos_enc(x)
        attentions = {'self': [], 'cross': []}
        for layer in self.layers:
            x, sa, ca = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            attentions['self'].append(sa)
            attentions['cross'].append(ca)
        x = self.norm(x)
        logits = self.fc_out(x)
        return logits, attentions

    def greedy_generate(self, memory, start_token_id, max_len=50, eos_token_id=None, device=None):
        device = device or memory.device
        B = memory.size(0)
        ys = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        for i in range(max_len):
            seq_len = ys.size(1)
            causal = generate_causal_mask(seq_len, device=device)
            logits, _ = self.forward(ys, memory, tgt_mask=causal, memory_mask=None)
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tokens], dim=1)
            if eos_token_id is not None and (next_tokens.squeeze(1) == eos_token_id).all():
                break
        print(ys)
        return ys
