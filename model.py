import torch
import torch.nn as nn
import numpy as np

D_MODEL = 512
D_FF = 2048
D_K = D_V = 64
N_LAYERS = 6
N_HEADS = 8


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 5000 * 512
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 5000 * 1
        div_term = pos / pow(
            10000.0, torch.arange(0, d_model, 2).float() / d_model
        )  # 5000 * 1 / 256 = 5000 * 256
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        pe = pe.unsqueeze(0)  # 1 * 5000 * 512

        self.register_buffer("pe", pe)

    def forward(self, x):
        pos_enc = self.pe[:, :x.size(1), :]
        x = x + pos_enc
        return self.dropout(x)


def get_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor):
    """
    seq_q: batch_size * len_q * d_model
    seq_k: batch_size * len_k * d_model

    seq_q * seq_k^T = len_q * len_k
    """

    batch_size, len_q = seq_q.size()
    len_k = seq_k.shape[1]

    pad_mask = torch.zeros((batch_size, len_q, len_k))

    for i in range(batch_size):
        for j in range(len_k):
            if seq_k[i, j] == 0:
                pad_mask[i, :, j] = 1

    pad_mask = pad_mask.expand(batch_size, len_q, len_k)
    pad_mask = pad_mask.to(torch.bool)

    return pad_mask


def get_subsequent_mask(seq: torch.Tensor):
    batch_size, word_num = seq.size()
    return torch.triu(torch.ones((batch_size, word_num, word_num)), diagonal=1).to(torch.bool)


def scaled_dot_production_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: torch.Tensor,
):
    # batch_size * n_headers * len_q * len_k
    scores: torch.Tensor = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_K)
    scores.masked_fill_(attn_mask, -1e9)
    attn = nn.Softmax(dim=-1)(scores)
    # (batch_size * n_headers * len_q * len_k) * (batch_size * n_headers * len_k *(d_model/n_heads))
    # -> batch_size * n_headers * len_q * (d_model/n_heads)
    context = torch.matmul(attn, V)
    return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(D_MODEL, D_MODEL)
        self.W_K = nn.Linear(D_MODEL, D_MODEL)
        self.W_V = nn.Linear(D_MODEL, D_MODEL)
        self.concat = nn.Linear(D_MODEL, D_MODEL)
        self.layer_norm = nn.LayerNorm(D_MODEL)

    def forward(
            self,
            Q: torch.Tensor,  # batch_size * len_q * d_model
            K: torch.Tensor,  # batch_size * len_k * d_model
            V: torch.Tensor,  # batch_size * len_k * d_model
            attn_mask: torch.Tensor,
    ):
        residual, batch_size = Q, Q.shape[0]

        # batch_size * len_q * n_headers * (d_model/n_heads) -> batch_size * n_headers * len_q * (d_model/n_heads)
        Q = (
            torch.Tensor(self.W_Q(Q))
            .view(batch_size, -1, N_HEADS, D_MODEL // N_HEADS)
            .transpose(1, 2)
        )

        # batch_size * len_k * n_headers * (d_model/n_heads) -> batch_size * n_headers * len_k * (d_model/n_heads)
        K = (
            torch.Tensor(self.W_K(K))
            .view(batch_size, -1, N_HEADS, D_MODEL // N_HEADS)
            .transpose(1, 2)
        )

        # batch_size * len_k * n_headers * (d_model/n_heads) -> batch_size * n_headers * len_k * (d_model/n_heads)
        V = (
            torch.Tensor(self.W_V(V))
            .view(batch_size, -1, N_HEADS, D_MODEL // N_HEADS)
            .transpose(1, 2)
        )

        attn_mask = attn_mask.unsqueeze(1).repeat(1, N_HEADS, 1, 1)
        context = scaled_dot_production_attention(Q, K, V, attn_mask)
        context = torch.cat(
            [context[:, head_num, :, :] for head_num in range(context.size(1))], dim=-1
        )
        output = self.concat(context)

        return self.layer_norm(output + residual)


class PosWiseFFN(nn.Module):
    def __init__(self):
        super(PosWiseFFN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, D_FF), nn.ReLU(), nn.Linear(D_FF, D_MODEL)
        )
        self.layer_norm = nn.LayerNorm(D_MODEL)

    def forward(self, inputs: torch.Tensor):
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.pos_wise_ffn = PosWiseFFN()

    def forward(self, enc_inputs: torch.Tensor, self_attn_mask: torch.Tensor):
        enc_outputs = self.self_attn(enc_inputs, enc_inputs, enc_inputs, self_attn_mask)
        enc_outputs = self.pos_wise_ffn(enc_outputs)
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, src_vocab_size: int):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, D_MODEL)
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])

    def forward(self, enc_inputs: torch.Tensor):
        enc_outputs: torch.Tensor = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        self_attn_mask = get_pad_mask(enc_inputs, enc_inputs)

        for layer in self.layers:
            enc_outputs = layer(enc_outputs, self_attn_mask)

        return enc_outputs


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.co_self_attn = MultiHeadAttention()
        self.pos_wise_ffn = PosWiseFFN()

    def forward(
            self,
            dec_inputs: torch.Tensor,
            enc_outputs: torch.Tensor,
            self_attn_mask: torch.Tensor,
            co_self_attn_mask: torch.Tensor,
    ):
        dec_outputs = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        dec_outputs = self.co_self_attn(
            dec_outputs, enc_outputs, enc_outputs, co_self_attn_mask
        )
        dec_outputs = self.pos_wise_ffn(dec_outputs)
        return dec_outputs


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size: int):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, D_MODEL)
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])

    def forward(
            self,
            dec_inputs: torch.Tensor,
            enc_inputs: torch.Tensor,
            enc_outputs: torch.Tensor,
    ):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        pad_mask = get_pad_mask(dec_inputs, dec_inputs)
        sub_mask = get_subsequent_mask(dec_inputs)
        self_attn_mask = torch.gt((pad_mask + sub_mask), 0)

        co_attn_mask = get_pad_mask(dec_inputs, enc_inputs)

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, self_attn_mask, co_attn_mask)

        return dec_outputs


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size)
        self.decoder = Decoder(tgt_vocab_size)
        self.projection = nn.Linear(D_MODEL, tgt_vocab_size)

    def forward(self, enc_inputs: torch.Tensor, dec_inputs: torch.Tensor):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1))
