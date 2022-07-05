from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from data_utils import AmazonDataset


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, args):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = args.embed_size
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device
        self.l2_lambda = args.l2_lambda
       
        self.w1=nn.Linear(args.embed_size,args.embed_size)
        self.w2=nn.Linear(args.embed_size,args.embed_size)
        self.w3=nn.Linear(args.embed_size,args.embed_size)
        self.w4=nn.Linear(args.embed_size,args.embed_size)
        self.w5=nn.Linear(args.embed_size,args.embed_size)
        self.w6=nn.Linear(args.embed_size,args.embed_size)
        self.w7=nn.Linear(args.embed_size,args.embed_size)

        # Initialize entity embeddings.
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            product=edict(vocab_size=dataset.product.vocab_size),
            related_product=edict(vocab_size=dataset.related_product.vocab_size),
            brand=edict(vocab_size=dataset.brand.vocab_size),
            category=edict(vocab_size=dataset.category.vocab_size),
        )
        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        self.relations = edict(
            purchase=edict(
                et='product',
                et_distrib=self._make_distrib(dataset.review.product_uniform_distrib)),
            produced_by=edict(
                et='brand',
                et_distrib=self._make_distrib(dataset.produced_by.et_distrib)),
            belongs_to=edict(
                et='category',
                et_distrib=self._make_distrib(dataset.belongs_to.et_distrib)),
            also_bought=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.also_bought.et_distrib)),
            also_viewed=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.also_viewed.et_distrib)),
            bought_together=edict(
                et='related_product',
                et_distrib=self._make_distrib(dataset.bought_together.et_distrib)),
            co_occr=edict(
                et='product',
                et_distrib=self._make_distrib(dataset.co_occr.et_distrib)),
        )
        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)  
        embed.weight = nn.Parameter(weight)   
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        batch_idxs: batch_size * 8 array, where each row is
                (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        user_idxs = batch_idxs[:, 0]
        product_idxs = batch_idxs[:, 1]
        brand_idxs = batch_idxs[:, 2]
        category_idxs = batch_idxs[:, 3]
        rproduct1_idxs = batch_idxs[:, 4]
        rproduct2_idxs = batch_idxs[:, 5]
        rproduct3_idxs = batch_idxs[:, 6]
        rproduct4_idxs = batch_idxs[:, 7]

        regularizations = []
        loss = torch.tensor(0.0)
        # user + purchase -> product
        up_loss, up_embeds = self.neg_loss('user', 'purchase', 'product', user_idxs, product_idxs,1)
        if up_loss is not None:
            regularizations.extend(up_embeds)
            loss += up_loss


        # product + produced_by -> brand
        pb_loss, pb_embeds = self.neg_loss('product', 'produced_by', 'brand', product_idxs, brand_idxs,2)
        if pb_loss is not None:
            regularizations.extend(pb_embeds)
            loss += pb_loss

        # product + belongs_to -> category
        pc_loss, pc_embeds = self.neg_loss('product', 'belongs_to', 'category', product_idxs, category_idxs,3)
        if pc_loss is not None:
            regularizations.extend(pc_embeds)
            loss += pc_loss
        
        #print("alsob")
 #       print("product_idxs-----------\n",product_idxs)
       # print("rproduct1_idxs-----------\n",rproduct1_idxs)
        # product + also_bought -> related_product1
        pr1_loss, pr1_embeds = self.neg_loss('product', 'also_bought', 'related_product', product_idxs, rproduct1_idxs,4)
        if pr1_loss is not None:
            regularizations.extend(pr1_embeds)
            loss += pr1_loss
        
      #  print("alsovvvvvvvvvv")
        # product + also_viewed -> related_product2
        pr2_loss, pr2_embeds = self.neg_loss('product', 'also_viewed', 'related_product', product_idxs, rproduct2_idxs,5)
        if pr2_loss is not None:
            regularizations.extend(pr2_embeds)
            loss += pr2_loss
        
      #  print("bttttttttttttt")
        # product + bought_together -> related_product3
        pr3_loss, pr3_embeds = self.neg_loss('product', 'bought_together', 'related_product', product_idxs, rproduct3_idxs,6)
        if pr3_loss is not None:
            regularizations.extend(pr3_embeds)
            loss += pr3_loss

      #  print("44444444444")
        # product + co_occr -> related_product4
        pr4_loss, pr4_embeds = self.neg_loss('product', 'co_occr', 'product', product_idxs,
                                                 rproduct4_idxs,7)
        if pr4_loss is not None:
            regularizations.extend(pr4_embeds)
            loss += pr4_loss

        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs,k):

        mask = (entity_tail_idxs >= 0) & (entity_head_idxs >= 0)
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []

        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding


            
                    
        #print("111111111111",entity_tail_embedding)
        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return kg_neg_loss(self,entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib,k)


def kg_neg_loss(self,entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib,k):
    """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

    Args:
        entity_head_embed: Tensor of size [vocab_size+1, embed_size].
        entity_tail_embed: Tensor of size [vocab_size+1, embed_size].
        entity_head_idxs:
        entity_tail_idxs:
        relation_vec: Parameter of size [1, embed_size].
        relation_bias: Tensor of size [batch_size]
        num_samples: An integer.
        distrib: Tensor of size [vocab_size].

    Returns:
        A tensor of [1].
    """
    batch_size = entity_head_idxs.size(0)
    entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
    entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]

               
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]
    
    
    #print("entity_tail_idxs-------------",entity_tail_idxs)

    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]

