import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import pickle
import torch

PRODUCT="product"

class GRU4REC(nn.Module):
    """
    d_model - the number of expected features in the input
    nhead - the number of heads in the multiheadattention models
    dim_feedforward - the hidden dimension size of the feedforward network model
    """
    def __init__(self,embedding_dim,hidden_dim,n_items):
        super(GRU4REC,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim =hidden_dim
        self.batch_first =True
        self.n_items =n_items
        self.pad_token = 0
    
        
       # self.move_embedding = nn.Embedding(n_items,embedding_dim,padding_idx=0)
        self.emb_pkl = pickle.load(open('tmp/Amazon_Beauty/transe_embed.pkl', 'rb'))
        self.move_embedding = nn.Embedding(n_items,embedding_dim,padding_idx=0)
        self.move_embedding.weight = torch.nn.Parameter(torch.tensor(self.emb_pkl[PRODUCT]))

        self.encoder_layer = nn.GRU(embedding_dim,self.hidden_dim,batch_first=self.batch_first)

        self.output_layer = nn.Linear(hidden_dim,hidden_dim)
    
    def forward(self,x,x_lens):
        x = self.move_embedding(x)
                    
        x = pack_padded_sequence(x,x_lens,enforce_sorted=False)

        output_packed,_ = self.encoder_layer(x)        
        x, _ = pad_packed_sequence(output_packed, batch_first=self.batch_first,padding_value=self.pad_token)
        
        x = self.output_layer(torch.sum(x, 1))
        return x
