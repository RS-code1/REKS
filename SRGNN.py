import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from torch_geometric.nn import GatedGraphConv
from utils import *

PRODUCT="product"

class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, all_item_embedding, batch):
        sections = torch.bincount(batch)
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        
        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        # Eq(8)
        # z_i_hat = torch.mm(s_h, all_item_embedding.weight.transpose(1, 0))
        
        return s_h



class SRGNN(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node):
        super(SRGNN, self).__init__()
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
        return self.e2s(hidden2, self.embedding, batch)