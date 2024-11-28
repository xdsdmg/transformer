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

        self.pe = torch.zeros(max_len, d_model)  # 5000 * 512
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 5000 * 1
        div_term = pos / pow(
            10000.0, torch.arange(0, d_model, 2).float() / d_model
        )  # 5000 * 1 / 256 = 5000 * 256
        self.pe[:, 0::2] = torch.sin(div_term)
        self.pe[:, 1::2] = torch.cos(div_term)
        self.pe = self.pe.unsqueeze(0)

        self.register_buffer("pe", self.pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


def get_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor):
    """
    seq_q: batch_size * len_q * d_model
    seq_k: batch_size * len_k * d_model

    seq_q * seq_k^T = len_q * len_k
    """

    batch_size, len_q, _ = seq_q.size()
    len_k = seq_k.shape[1]

    pad_mask = torch.ones((batch_size, len_q, len_k))

    for i in range(batch_size):  # TODO: need confirmed
        for j in range(len_k):
            r = torch.abs(seq_k[i, j, :])
            if torch.sum(r) != 0:
                pad_mask[i, :, j] = 0

    return pad_mask.expand(batch_size, len_q, len_k)


def get_subsequent_mask(seq: torch.LongTensor):
    batch_size, l, _ = seq.size()
    return torch.triu(torch.ones((batch_size, l, l)), diagonal=1).to(torch.bool)


def scaled_dot_production_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor,
):

    # batch_size * n_headers * len_q * len_k
    scores: torch.LongTensor = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_K)
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


"""
编码器的主要功能是对输入序列进行特征提取和语义编码。它会将输入序列（例如在自然语言处理中，可能是一个句子）中的每个元素（单词或子词等）转换为一个包含语义信息的向量表示。以处理句子 “我喜欢阅读书籍” 为例，编码器会将每个词映射到一个高维向量空间，这个向量空间中存储了关于这个词的语义信息，同时也包含了它与句子中其他词的关系信息。From Doubao
"""


class Encoder(nn.Module):
    def __init__(self, src_vocab_size: int):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, D_MODEL)
        self.pos_emb = PositionalEncoding(D_MODEL)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])

    def forward(self, enc_inputs: torch.Tensor):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        self_attn_mask = get_pad_mask(enc_inputs, enc_inputs)

        for layer in self.layers:
            enc_outputs = layer(enc_outputs, self_attn_mask)

        return enc_outputs


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_wise_ffn = PosWiseFFN()
