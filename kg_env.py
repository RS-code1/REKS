from __future__ import absolute_import, division, print_function

import os
import sys
from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime
import numpy as np
import torch.nn as nn
import pandas as pd
import math
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.filterwarnings('ignore')

from knowledge_graph import KnowledgeGraph
from utils import *

class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len  # mode: one of {full, current}
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 4 * embed_size
        elif history_len == 2:
            self.dim = 6 * embed_size
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, user_embed, older_node_embed,older_relation_embed,last_node_embed,last_relation_embed, node_embed
                 ):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, last_node_embed, last_relation_embed,node_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed,older_node_embed,older_relation_embed, last_node_embed, last_relation_embed, node_embed ])
        else:
            raise Exception('mode should be one of {full, current}')


class BatchKGEnvironment(object):
    def __init__(self, dataset_str, max_acts, state_dim, max_path_len, state_history=1):
        super(BatchKGEnvironment, self).__init__()
        self.max_acts = max_acts
        self.act_dim = max_acts 
        self.max_num_nodes = max_path_len + 1 
        self.kg = load_kg(dataset_str)
        self.embeds = load_embed(dataset_str)
        #self.embed_size = self.embeds[USER].shape[1]     #user
        self.embed_size = self.embeds[PRODUCT].shape[1]       #last item
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_gen = KGState(self.embed_size, history_len=state_history)
        self.state_dim = self.state_gen.dim
        self.lstm_layer=torch.nn.LSTM(input_size=self.embed_size*2,hidden_size=self.embed_size*2,num_layers=1)
        #u_p_scores = np.dot(self.embeds[USER] + self.embeds[PURCHASE][0], self.embeds[PRODUCT].T)
        #self.u_p_scales = np.max(u_p_scores, axis=1)



        self._batch_path = None
        self._batch_curr_actions = None
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self._done = False


    def _get_actions(self, path, done):
        _, curr_node_type, curr_node_id = path[-1]
        actions = [(SELF_LOOP, curr_node_id, curr_node_type)]

        if done:
            return actions

        # 获得所有的邻居结点
        relations_nodes = self.kg(curr_node_type, curr_node_id)
        candidate_acts = []  # list of tuples of (relation, node_id)
        visited_nodes = set([(v[1], v[2]) for v in path])
        for r in relations_nodes:
            next_node_type = KG_RELATION[curr_node_type][r]
            
            next_node_ids = relations_nodes[r]
            next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]  # filter
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids,[next_node_type]*len(next_node_ids)))
            l=zip([r] * len(next_node_ids), next_node_ids,[next_node_type]*len(next_node_ids))
        if len(candidate_acts) == 0:
            return actions

        if len(candidate_acts)<=self.max_acts:
            return candidate_acts
        actions=random.sample(candidate_acts,self.max_acts)
        return actions


    def _batch_get_actions(self, batch_path, done):
        #print(batch_path[0])
        return [self._get_actions(path[1:], done) for path in batch_path]




    def _get_state(self, path):
        #print(path[0])
        #user_embed = self.embeds[USER][path[0][-1]]    #user
        user_embed = self.embeds[PRODUCT][path[0][-1]]     #last item
        zero_embed = np.zeros(self.embed_size)
        if len(path) == 1:
            state = self.state_gen(user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)
            return state
        older_relation, last_node_type, last_node_id = path[-2]
        last_relation, curr_node_type, curr_node_id = path[-1]
        curr_node_embed = self.embeds[curr_node_type][curr_node_id]
        last_node_embed = self.embeds[last_node_type][last_node_id]
        last_relation_embed, _ = self.embeds[last_relation]
        if len(path) == 2:
            state = self.state_gen(user_embed, last_node_embed, last_relation_embed,curr_node_embed, zero_embed,
                                   zero_embed)
            return state

        _, older_node_type, older_node_id = path[-3]
        older_node_embed = self.embeds[older_node_type][older_node_id]
        older_relation_embed, _ = self.embeds[older_relation]
        state = self.state_gen(user_embed, older_relation_embed,older_relation_embed,last_node_embed,last_relation_embed,curr_node_embed)
        return state
    

    def _batch_get_state(self, batch_path):
        batch_state = [self._get_state(path[1:]) for path in batch_path]
        return np.vstack(batch_state)  # [bs, dim]



       
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))


    def _get_reward(self, path, session_rep, target,batch_sessions,batch_actions):
        #print(path,batch_sessions)

        
        if len(path) < self.max_num_nodes:   #user 3  last item 2
            return 0.0

        score_item,score_path,target_score = 0.0,0.0,0.0
        score_step = 0.0 
        score_ndcg = 0.0 

        

        invalid_product = []
        for i in range(len(batch_actions)):
            if batch_actions[i][-1] != PRODUCT:
                continue
            invalid_product.append(batch_actions[i][-2])

        idx=np.where(invalid_product==int(target),1,0)
        c=np.arange(len(invalid_product))
        
        c=list(map(lambda x:1/math.log(x+2,2),c))
        #c=list(map(lambda x:1/(i+1),c))
        score_ndcg=sum(idx*c)
 
            
      
        _, curr_node_type, curr_node_id = path[-1]
          
        
        if curr_node_type == PRODUCT:
            if curr_node_id == int(target):
                score_item = 1.0
            else:
                p_vec = self.embeds[PRODUCT][curr_node_id]
                last_session_item=batch_sessions[-1]
                last_session_emb = self.embeds[PRODUCT][last_session_item]
                score_item=self.sigmoid(np.dot(p_vec, last_session_emb))

        # path reward
        path_embedding=np.zeros((len(path),self.embed_size))
        for i,(_, node_type, node_id) in enumerate(path):
            path_embedding[i,:]=self.embeds[node_type][node_id]
        path_embedding=np.mean(path_embedding, axis=0)
        score_path=self.sigmoid(np.dot(path_embedding, session_rep))

        target_score=2**score_ndcg+score_item+score_path

        #print("target_score=score_item+score_path",target_score,score_item,score_path,"\n")
        return target_score
#        return score_item



    def _batch_get_reward(self, batch_path,session_dict,targets,batch_sessions,batch_actions):
        batch_reward=[]

        for i in range(len(batch_path)):
            
            batch_reward.append(self._get_reward(batch_path[i][1:],session_dict[batch_path[i][0]],targets[i],batch_sessions[i],batch_actions[i]))
        #print("np.array(batch_reward)",np.array(batch_reward))
        return np.array(batch_reward)
        

    def _is_done(self):
        return self._done or len(self._batch_path[0]) >= self.max_num_nodes+1



    def reset(self,session_dict,sids,targets,batch_sessions,uids):   #last item
        if uids is None:
            all_uids = list(self.kg(PRODUCT).keys())
            uids = [random.choice(all_uids)]
        # each element is a tuple of (relation, entity_type, entity_id)
        self._batch_path = [[sids[i],(SELF_LOOP, PRODUCT, uids[i])] for i in range(len(uids))]
        self._done = False
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)
        self._batch_curr_reward = self._batch_get_reward(self._batch_path,session_dict,targets,batch_sessions,self._batch_curr_actions)

        return self._batch_curr_state

    def batch_step(self, batch_act_idx,session_dict,targets,batch_sessions,our_scores):
        """
        Args:
            batch_act_idx: list of integers.
			
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act_idx) == len(self._batch_path)

        for i in range(len(batch_act_idx)):
            act_idx = batch_act_idx[i]
            _, curr_node_type, curr_node_id = self._batch_path[i][-1]
            relation, next_node_id,_ = self._batch_curr_actions[i][act_idx]  
            if relation == SELF_LOOP:
                next_node_type = curr_node_type
            else:
                next_node_type = KG_RELATION[curr_node_type][relation]
            self._batch_path[i].append((relation, next_node_type, next_node_id))  
        #print("path",self._batch_path,len(self._batch_path))
        self._done = self._is_done()  # must run before get actions, etc.
        #print(self._done)
        self._batch_curr_state = self._batch_get_state(self._batch_path)  

        batch_actions=self._batch_curr_actions

        action_pro=[] 
              
        for j in range(len(self._batch_curr_actions)):
            action_pro=zip(our_scores[j],batch_actions[j]) 
            action_pro=sorted(action_pro,reverse=True)          
            pro_new,actions_new=zip(*action_pro)
            batch_actions[j]=actions_new

            
        
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done)  
        self._batch_curr_reward = self._batch_get_reward(self._batch_path,session_dict,targets,batch_sessions,batch_actions)  
        #print("3self._batch_curr_reward",self._batch_curr_reward)
        return self._batch_curr_state, self._batch_curr_reward, self._done


    def get_action_embedding(self,batch_path,batch_actions):
        contype='node'   ### relation/sum/concate/node
        action_embedding = np.zeros((len(batch_actions), self.act_dim, self.embed_size))
        for i in range(len(batch_actions)):
            for j in range(len(batch_actions[i])):
                #print("batch_actions[i][j]\n",batch_actions[i][j])
                relation, next_node_id, _= batch_actions[i][j]
                a, curr_node_type, curr_node_id = batch_path[i][-1]
                if relation == SELF_LOOP:
                    next_node_type = curr_node_type
                else:
                    next_node_type = KG_RELATION[curr_node_type][relation]

                if contype=='relation':
                    action_embedding[i, j, :] =self.embeds[relation][0]
                elif contype=='sum':
                    action_embedding[i, j, :] = self.embeds[relation][0] + self.embeds[next_node_type][next_node_id]
                elif contype=='node':
                    action_embedding[i, j, :] = self.embeds[next_node_type][next_node_id]
                elif contype=='concate':
                    action_embedding[i, j, :] = np.concatenate(
                        (self.embeds[relation][0], self.embeds[next_node_type][next_node_id]))
                else:    
                    if len(batch_path[i])>2:
                        st_embedding=np.zeros((len(batch_path[i])-2,self.embed_size*2))
                        
                        for ii in range(1,len(batch_path[i])-1):
                            cur_relation,node_type, node_id = batch_path[i][ii]
                            
                            cur_relation_emb=self.embeds[cur_relation][0]
                            node_emb=self.embeds[node_type][node_id]
                            st_embedding[ii-1,:]=np.hstack([cur_relation_emb,node_emb])

                        st='lstm'
                        if st=='mean':
                            #mean
                            st_embedding=np.mean(st_embedding, axis=0)
                        else:
                            #lstm  
                            
                            lstm_input=torch.tensor(st_embedding,dtype=torch.float).unsqueeze(0)
                            output,(h_n,c_n)=self.lstm_layer(lstm_input)
                            st_embedding=output[0][-1].detach().numpy()

                        
                        action_embedding[i, j, :]=(action_embedding[i, j, :]+st_embedding)/2              
                    

        return action_embedding

    def batch_action_mask(self, kz,dropout=0.0):
        batch_mask = []
        for actions in self._batch_curr_actions:
           # if kz==0:
           #     print("alenn",len(actions))
            act_idxs = list(range(len(actions)))
            if dropout > 0 and len(act_idxs) > 10:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(act_idxs[1:], keep_size, replace=False).tolist()
                act_idxs = [act_idxs[0]] + tmp
            act_mask = np.zeros(self.act_dim, dtype=np.bool)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        return np.vstack(batch_mask)

    def print_path(self):
        for path in self._batch_path:
            msg = 'Path: {}({})'.format(path[0][1], path[0][2])
            for node in path[1:]:
                msg += ' =={}=> {}({})'.format(node[0], node[1], node[2])
            print(msg)

