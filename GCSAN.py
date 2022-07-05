from cmath import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from torch_geometric.nn import GatedGraphConv
from utils import *


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 400
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dp = nn.Dropout(p=0.2)
        self.drop = True

    def forward(self, x):
        residual = x  # keep original input
        x = F.relu(self.d1(x))
        if self.drop:
            x = self.d2(self.dp(x))
        else:
            x = self.d2(x)
        out = residual + x
        return out


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.rn = Residual()
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 1)

    def forward(self, session_embedding, batch):

        sections = torch.bincount(batch)
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        hidden=torch.cat(v_i).reshape(max(batch)+1,-1,session_embedding.shape[1])
        #self attention
        k_blocks=4
        ht = hidden[torch.arange(hidden.shape[0]).long(), sections - 1]  # batch_size x latent_size

        # 加上 self attention
        attn_output = hidden
        for k in range(k_blocks):
            attn_output = attn_output.transpose(0,1)
            attn_output, attn_output_weights = self.multihead_attn(attn_output, attn_output, attn_output)

            # attn_output = self.multihead_attn(attn_output, attn_output, attn_output, mask_self)  # 加上mask
            attn_output = attn_output.transpose(0,1)

            attn_output = self.rn(attn_output)
        hn = attn_output[torch.arange(hidden.shape[0]).long(), sections - 1]  # use last one as global interest
        # a = hn + ht  # consider current interest
        a = 0.52*hn + (1-0.52)*ht  # hyper-parameter w
        return a

class GCSAN(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node):
        super(GCSAN, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node

        self.emb_pkl = pickle.load(open('tmp/Amazon_Beauty/transe_embed.pkl', 'rb'))
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)
        self.embedding.weight = torch.nn.Parameter(torch.tensor(self.emb_pkl[PRODUCT]))

        self.e2s = Embedding2Score(self.hidden_size)
        self.gated = GatedGraphConv(self.hidden_size, num_layers=1)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index,batch):
        embedding = self.embedding(x.to(self.device)).squeeze()
        hidden = self.gated(embedding, edge_index.to(self.device))
        hidden2 = F.relu(hidden)
        return self.e2s(hidden2, batch)