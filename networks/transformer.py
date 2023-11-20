import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import sys




class TransformerEncoder(torch.nn.Module):
    """Encoder network for Transformer"""
    def __init__(self, d_model, n_layers, num_heads, embs_npa, max_seq_length, freeze_embeddings=True):
        super().__init__()
        self.n_layers = n_layers

        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embs_npa).float(), 
            freeze=freeze_embeddings
        )

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoding_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for i in range(n_layers)])

        self.norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, inputs, training=True):
        # seq_length = torch.shape(inputs)[1]

        x = self.embedding_layer(inputs)
        x = self.positional_encoding(x)

        for i in range(self.n_layers):
            x = self.encoding_layers[i](x)
        
        return self.norm(x)


# consider defining own Norm class
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model, dropout)
        self.ff = FFN(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # x2 = self.norm_1
        attn_output, _ = self.multi_head_attention([x, x, x, mask])
        attn_output = self.dropout_1(attn_output)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ff(out1)
        ffn_output = self.dropout_2(ffn_output)
        out2 = self.norm_2(out1 + ffn_output)

        return out2


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # d_k for splitting heads
        self.d_model = d_model

        self.query_dense = nn.Linear(d_model)
        self.key_dense = nn.Linear(d_model)
        self.value_dense = nn.Linear(d_model)

        # self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

        self.out = nn.Linear(d_model, d_model)

    
    def scaled_dot_prod(self, q, k, v, mask=None):
        """scaled dot product, where query, key, and value are vectors"""
        q_matmul_k = torch.matmul(q, k.transpose(-2, -1))

        scaled_attention_logits = q_matmul_k / torch.sqrt(self.depth)  # d_k

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  # effectively ignore the masked values

        attention_weights = nn.Softmax(scaled_attention_logits, dim=-1)  # scores

        # consider applying dropout here

        output = torch.mm(attention_weights, v)

        return output, attention_weights


    def split_heads(self, inputs, batch_size):
        inputs = torch.reshape(inputs, (batch_size, -1, self.num_heads, self.depth))
        # transpose to get dimensions bs * h * sl * d_model
        return inputs.transpose(1, 2)  # check

    
    def call(self, inputs):
        query, key, value, mask = inputs
        batch_size = query.shape[0]

        # multiple input vectors times q, k, v matrics to get q, k, v, vectors
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(mask)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_prod(q=query, k=key, v=value, mask=mask)
        scaled_attention = scaled_attention.transpose(1,2).contiguous()
        
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.out(concat_attention)

        return output, attention_weights



class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        # perturb each embedding in some random way

        pe = torch.zeros(max_seq_length, d_model)  # positional encoding
        for pos in range(max_seq_length):
            # map every other one to sin or cos
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))

        pe = pe.unsqueeze(0)  # add dimension of size 1 at index 0
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x *= math.sqrt(self.d_model)

        # add constant to embedding
        seq_length = x.size(1)

        x += Variable(self.pe[:,:seq_length], requires_grad=False)  # .cuda()

        return x


class FFN(torch.nn.Module):
    """Feed Forward Network"""
    def __init__(self, d_model, hidden_size=2048, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_size, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
