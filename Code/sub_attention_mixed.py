import torch
import torch.nn as nn

feature_bins = 512
d_ff = 512 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention

def init_gobal_variable(a=512,b=512,c=64,d=8):
    global feature_bins
    global d_ff
    global d_k
    global d_v
    global n_heads
    feature_bins = a
    d_ff = b
    d_k = d_v = c
    n_heads = d

def ScaledDotProductAttention( Q, K, V):
    scores = torch.matmul(Q, K.transpose(-1, -2))
    attn = nn.Softmax(dim=-1)(scores)
    context = torch.matmul(attn, V)
    return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.dk = d_k
        self.dv = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(feature_bins, d_k * n_heads)
        self.W_K = nn.Linear(feature_bins, d_k * n_heads)
        self.W_V = nn.Linear(feature_bins, d_v * n_heads)
        self.output = nn.Linear(n_heads * d_v, feature_bins)
        self.layerNorm = nn.LayerNorm(feature_bins)

    def forward(self, Q, K, V):
        # q: [batch_size x len_q x d_model],
        # k: [batch_size x len_k x d_model],
        # v: [batch_size x len_k x d_model]

        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1,  self.n_heads, self.dk).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1,  self.n_heads, self.dk).transpose(1,2)
        # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1,  self.n_heads, self.dv).transpose(1,2)
        # v_s: [batch_size x n_heads x len_k x d_v]

        context, attn = ScaledDotProductAttention(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,  self.n_heads * self.dv)

        output = self.output(context)

        residual_ouput = self.layerNorm(output + residual)

        return residual_ouput, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_bins, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=feature_bins, kernel_size=1)
        self.layerNorm = nn.LayerNorm(feature_bins)

    def forward(self, inputs):
        # enc_outputs: torch.Size([1, 5, 512])
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        # print(inputs.transpose(1, 2).size())
        # torch.Size([1, 512, 5])
        # print('self.conv1(inputs.transpose(1, 2)):', self.conv1(inputs.transpose(1, 2)).size())
        # self.conv1(inputs.transpose(1, 2)): torch.Size([1, 2048, 5])
        output = self.conv2(output).transpose(1, 2)
        # (1, 512, 5)---(1, 5, 512)

        residual_ouput = self.layerNorm(output + residual)

        return residual_ouput


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, Q, K, V):
        # enc_outputs: torch.Size([1, 5, 512])
        # enc_self_attn_mask: torch.Size([1, 5, 5])
        enc_outputs, attn = self.enc_self_attn(Q, K, V)
        # output: torch.Size([1, 5, 512])
        # attn: torch.Size([1, 8, 5, 5])

        # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, Encoder_n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(Encoder_n_layers)])

    def forward(self, q,data):
        # 假设输入特征为 （Batch, Time, log_mel bins）
        # enc_inputs: torch.Size([16, 31, 527])

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(q,data,data)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs

# if __name__ == '__main__':
#     init_gobal_variable(233)
#     model = Encoder(1)
#     input = torch.rand(16,31,233)
#     x= model(input)
#     print(x.size())
    # print(y[0].size())