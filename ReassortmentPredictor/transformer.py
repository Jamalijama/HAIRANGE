import torch
import torch.nn as nn
import numpy as np

from datasets import *

# d_model = 1024  # Dimension of word embedding
# d_ff = 2048  # the dimensions of the forward propagation hide layer
# d_k = d_v = 64  # Dimension of K(=Q), V
# n_layers = 6  # the layers of encoder and decoder
# n_heads = 8  # the head number of Multi-Head Attention

d_model = 64  # Dimension of word embedding
d_ff = 128  # the dimensions of the forward propagation hide layer
d_k = d_v = 4  # Dimension of K(=Q), V
n_layers = 6  # the layers of encoder and decoder
n_heads = 1  # the head number of Multi-Head Attention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=1000):
        # dropout=0.1
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # The word embedding dimension is even
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # The word embedding dimension is odd
        self.pos_table = torch.FloatTensor(pos_table).to(device)  # enc_inputs: [batch_size, seq_len, d_model]

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        # enc_inputs = enc_inputs.to(device)
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        # del self.pos_table
        return self.dropout(enc_inputs.to(device))


def get_attn_pad_mask(seq_q, seq_k):  # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # tensor.data returns the same tensor, and this new tensor shares data with the old tensor
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # Determine which inputs contain P(=0) and mark them with 1,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k).to(device)  # Expand into multiple dimensions


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, len_q, len_k]

        scores.masked_fill_(attn_mask, 0)  # If it's a stop word P is equal to 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, fixed_value):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.W_Q.weight.data.fill_(fixed_value)
        self.W_K.weight.data.fill_(fixed_value)
        self.W_V.weight.data.fill_(fixed_value)
        self.fc.weight.data.fill_(fixed_value)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 0).reshape(batch_size, -1, n_heads * d_v)
        # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, fixed_value):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            # nn.BatchNorm1d(d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False), )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                layer.weight.data.fill_(fixed_value)
        # nn.BatchNorm1d(d_model))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, fixed_value):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(fixed_value)  # Multi-head attention mechanism
        self.pos_ffn = PoswiseFeedForwardNet(fixed_value)  # feedforward neural network

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        # Input 3 enc_inputs and multiply them with W_Q, W_K and W_V respectively to get Q, K and V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size, src_len, d_model]
        # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs.to(device), attn


class Encoder(nn.Module):
    def __init__(self, fixed_value):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)  # Add location information
        self.layers = nn.ModuleList([EncoderLayer(fixed_value) for _ in range(n_layers)])

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, src_len, d_model]
        # enc_outputs = self.src_emb(enc_inputs)  # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_inputs).to(device)  # enc_outputs: [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs[:, :, 0], enc_inputs[:, :, 0])
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # enc_outputs : [batch_size, src_len, d_model],
            # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs.to(device), enc_self_attns


class Transformer(nn.Module):
    def __init__(self, fixed_value=0.1):
        super(Transformer, self).__init__()
        self.Encoder = Encoder(fixed_value).to(device)
        self.norm = nn.LayerNorm(d_model)
        # self.projection = nn.Linear(d_model, 3, bias=False).to(device)

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)
        # enc_outputs: [batch_size, src_len, d_model]
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # enc_outputs = enc_outputs.mean(dim=1)
        enc_outputs = self.norm(enc_outputs)
        # enc_outputs = self.projection(enc_outputs)
        # enc_outputs = torch.sigmoid_(enc_outputs)
        return enc_outputs.to(device)
