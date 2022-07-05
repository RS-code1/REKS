from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import numpy as np
from itertools import chain
import pandas as pd
import pickle
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from NARM import NARM
from SRGNN import SRGNN
from GRU4REC import GRU4REC
from GCSAN import GCSAN
from bert_modules.bert import BERT
from utils import *
import math

logger = None

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
base_model='narm' #srgnn/narm/gru4rec/GCSAN/bert4rec

class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim,gamma=0.99, hidden_sizes=400):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        
        self.l1 = nn.Linear(400, hidden_sizes)
        #self.l2 = nn.Linear(200, 100)
        self.l2 = nn.Linear(400+state_dim, hidden_sizes)
        #self.l3 = nn.Linear(200, 100)
        self.l3 = nn.Linear(state_dim, 400)  #lstm gru
        self.critic = nn.Linear(hidden_sizes, 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs,narm_state,act_embedding,kz):
        #narm   st
        
        #print(act_embedding.shape)
        
        #session_narm  st
        state, act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        
        
        # print(act_embedding.shape)
        x = self.l1(act_embedding)
        x = F.dropout(F.elu(x), p=0.5)


 #       out1 = self.l3(state)
 #       x1 = F.dropout(F.elu(out1), p=0.5)


        #state_values = self.critic(state)
        
        x1=torch.hstack([state,narm_state])
        out = self.l2(x1)
        x1 = F.dropout(F.elu(out), p=0.5)

        state_values = self.critic(x1)

        actor_logits = torch.matmul(x, x1.unsqueeze(2)).squeeze(2)

        actor_logits[1-act_mask.long()] = -999999.0  
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]
        # print(self.l3.weight)
        return act_probs,x1,state_values


        
    def select_action(self, kz,batch_state,narm_state,action_embedding, batch_act_mask, session_dic,batch_path, device):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state = torch.FloatTensor(batch_state).to(device)  # Tensor [bs, state_dim]
        #narm_state = torch.FloatTensor(session_rep).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.ByteTensor(batch_act_mask).to(device).bool()  # Tensor of [bs, act_dim]
        action_embedding=torch.FloatTensor(action_embedding).to(device)  # Tensor of [bs, act_dim, emb_size*2]
        probs,x1,state_values=self((state, act_mask), narm_state,action_embedding,kz)
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ]


        
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), state_values))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist(),probs,x1

    def update(self):
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]




class ACDataLoader(object):
    def __init__(self, data, batch_size):
        ## data:dataframe uid:user_id session:list  target:item_id
        self.uids = np.array(data['user'].unique())
        self.data=data
        self.num_sessions = len(data)
        self.batch_size = batch_size
        self.reset()

    def reset(self):

        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self,kz):
        if not self._has_next:
            return None

        end_idx = min(self._start_idx + self.batch_size, self.num_sessions)

        batch_sessions = self.data['session'].iloc[self._start_idx:end_idx].tolist()
        #batch_uids=self.data['user'].iloc[self._start_idx:end_idx] 
        batch_uids=[x[-1] for x in batch_sessions]  ###起点为 last item
        batch_target=self.data['label'].iloc[self._start_idx:end_idx]
        batch_len=self.data['length'].iloc[self._start_idx:end_idx].tolist()
        batch_sids = self.data['session_id'].iloc[self._start_idx:end_idx].tolist()
        padded_sesss_item = torch.zeros(end_idx-self._start_idx, max(batch_len)).long()
        #print('==================================',len(padded_sesss_item))
        for i,se in enumerate(batch_sessions):
            padded_sesss_item[i, :batch_len[i]] = torch.LongTensor(se)
        #print(len(padded_sesss_item))
        self._has_next = self._has_next and end_idx < self.num_sessions
        self._start_idx = end_idx
        #print(padded_sesss_item)
        #print(padded_sesss_item.transpose(0,1))
        #return padded_sesss_item.transpose(0,1),batch_len, torch.tensor(batch_target.tolist()).long(),batch_uids.tolist(),batch_sids  
        return padded_sesss_item.transpose(0, 1), batch_len, torch.tensor(batch_target.tolist()).long(), batch_uids, batch_sids  


def train(args):
    
    path='./data/'
    train_data=pd.read_csv(path+'train_session.csv')
    train_data['session'] = train_data['session'].apply(lambda x: eval(x))
    train_data['length'] = train_data['session'].apply(lambda x: len(x))
    train_data.sort_values(by=['length'], inplace=True, ascending=False)
    train_data.index=np.arange(len(train_data))
    train_data['session_id'] = np.arange(len(train_data))
    train_labels = load_labels(args.dataset, 'train')
    test_data = pd.read_csv(path + 'test_session.csv')
    test_data['session'] = test_data['session'].apply(lambda x: eval(x))
    test_data['length'] = test_data['session'].apply(lambda x: len(x))
    test_data.sort_values(by=['length'], inplace=True, ascending=False)
    test_data.index = np.arange(len(test_data))
    test_data['session_id'] = np.arange(len(test_data))
    test_labels = dict(zip(list(test_data['session_id']), list(test_data['label'])))
    session_user_dict = dict(zip(list(test_data['session_id']), list(test_data['user'])))

    n_items=len(np.load(path+'item_dict.npy',allow_pickle=True).item())
    print('n_items',n_items)
    #print('item',n_items)
    env = BatchKGEnvironment(args.dataset, args.max_acts, args.state_dim, max_path_len=args.max_path_len, state_history=args.state_history)
    if base_model=='srgnn':
        session=SRGNN(args.embed_dim,n_items).to(args.device)
    elif base_model=='narm':
        session=NARM(n_items, int(env.embed_size/2), args.embed_dim, args.batch_size).to(args.device)
    elif base_model=='gru4rec':
        session=GRU4REC(args.embed_dim,args.embed_dim,n_items).to(args.device)
    elif base_model=='GCSAN':
        session=GCSAN(args.embed_dim,n_items).to(args.device)
    elif base_model=='bert4rec':
        session=BERT(args).to(args.device)

    train_dataloader = ACDataLoader(train_data, args.batch_size)
    test_dataloader=ACDataLoader(test_data, 100)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(params=chain(session.parameters(), model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    total_losses, total_plosses, total_entropy_act, total_entropy, total_rewards = [], [], [], [], []
    step = 0

    session_dic = {}
    last_hr = [0.0,0.0,0.0]
    last_mrr = [0.0,0.0,0.0]
    last_ndcg = [0.0,0.0,0.0]
    # test(args,model,env,session,test_dataloader,session_user_dict,train_labels,test_labels)
    for epoch in range(1, args.epochs + 1):
        ### Start epoch ###
        kz=0
        #print("----------",epoch)
        
        #env.train()
        session.train()
        train_dataloader.reset()
        #print("args.epochs",args.epochs,epoch)
        while train_dataloader.has_next():
            optimizer.zero_grad()
            batch_sessions,seq_length,targets,batch_uids,batch_sids= train_dataloader.get_batch(kz)
            
            ### Start batch episodes ###

            if base_model=='srgnn' or base_model=='GCSAN':
                sequences=batch_sessions.transpose(0,1)
                x = []
                batch=[]
                senders,receivers=[],[]
                i,j=0,0
                for sequence in sequences:
                    sender = []
                    nodes = {}
                    for node in sequence:
                        if node not in nodes:
                            nodes[node] = i
                            x.append([node])
                            i += 1
                        batch.append(j)
                        sender.append(nodes[node])
                    j += 1
                    receiver = sender[:]
                    if len(sender) != 1:
                        del sender[-1]
                        del receiver[0]
                    senders.extend(sender)
                    receivers.extend(receiver)
                edge_index = torch.tensor([senders, receivers], dtype=torch.long)
                x = torch.tensor(x, dtype=torch.long)
                batch = torch.tensor(batch, dtype=torch.long)
                session_rep=session(x,edge_index,batch)
            
            elif base_model=='narm':
                session_rep,score=session(batch_sessions.to(args.device),seq_length) #[bs,embed_dim]
            elif base_model=='gru4rec':
                session_rep=session(batch_sessions.to(args.device),seq_length) #[bs,embed_dim]
            elif base_model=='bert4rec':
                if batch_sessions.shape[0]<5:
                    batch_sessions=torch.cat((batch_sessions,torch.zeros(5-batch_sessions.shape[0],batch_sessions.shape[1],dtype=torch.long)))
                session_rep=session(batch_sessions.transpose(0,1).to(args.device))
            
            batch_sessions=batch_sessions.transpose(0,1)
            dict1 = dict(zip(batch_sids, session_rep.tolist()))
            session_dic.update(dict1)

            batch_state = env.reset(session_dic,batch_sids,targets, batch_sessions,batch_uids)  # numpy array of [bs, state_dim]


            done = False
            our_scores,batch_curr_actions = [],[]
            while not done:
                batch_act_mask = env.batch_action_mask(kz,dropout=args.act_dropout)  # numpy array of size [bs, act_dim]   action drop
                action_embedding = env.get_action_embedding(env._batch_path, env._batch_curr_actions)


                batch_act_idx,our_scores,s_emb = model.select_action(kz,batch_state,session_rep, action_embedding, batch_act_mask, session_dic,env._batch_path, args.device)  # int   ##选择action
                #print("train_batch_step")
                kz=1
                batch_state, batch_reward, done = env.batch_step(batch_act_idx,session_dic,targets,batch_sessions,our_scores)

                batch_curr_actions=env._batch_curr_actions

                
                model.rewards.append(batch_reward)
                
            ### End of episodes ###
            
            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(train_data) / args.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr



            total_rewards.append(np.sum(model.rewards))


            
            p_emb=model.l1(torch.tensor(env.embeds[PRODUCT]).to(args.device))
            scores=torch.matmul(s_emb,p_emb.T)

            loss, ploss, ealoss, eloss = get_loss(model.rewards,model.saved_actions,model.entropy,scores.to(args.device),targets.to(args.device),args.gamma,args.ent_weight,criterion)


            loss.backward()
            
            optimizer.step()
            model.update()

            total_losses.append(loss.item())
            total_plosses.append(ploss.item())
            total_entropy_act.append(ealoss.item())
            total_entropy.append(eloss.item())
            step += 1

            # Report performance
            if step > 0 and step % 200 == 0:
                avg_reward = np.mean(total_rewards) / args.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_entropy_act=np.mean(total_entropy_act)
                avg_entropy = np.mean(total_entropy)
                total_losses, total_plosses,total_entropy_act, total_entropy, total_rewards = [], [], [], [], []
                logger.info(
                    'epoch/step={:d}/{:d}'.format(epoch, step) +
                    ' | loss={:.5f}'.format(avg_loss) +
                    ' | ploss={:.5f}'.format(avg_ploss) +
                    ' | ealoss={:.5f}'.format(avg_entropy_act) +
                    ' | entropy={:.5f}'.format(avg_entropy) +
                    ' | reward={:.5f}'.format(avg_reward))
            ### END of epoch ###


        policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.log_dir, epoch)
        logger.info("Save model to " + policy_file)
        torch.save({'model': model.state_dict(), 'narm': session.state_dict()}, policy_file)
    
        best_hr, best_mrr, best_ndcg = test(args,model,env,session,test_dataloader,session_user_dict,train_labels,test_labels)
        if best_mrr[0] > last_mrr[0]:
            last_hr = best_hr
            last_mrr = best_mrr
            last_ndcg = best_ndcg
        print("best_hr@5,10,20", [float("{:.4f}".format(i)) for i in last_hr], "\nbest_mrr@5,10,20", [float("{:.4f}".format(i)) for i in last_mrr],"\nbest_ndcg@5,10,20", [float("{:.4f}".format(i)) for i in last_ndcg])

def evaluate(topk_matches, target,k):
    
    
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    test_user_products=target

    
    invalid_users = []
    # Compute metrics
    hrs, ndcgs,mrrs = [], [], []
    
    
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < k:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid], test_user_products[uid]
        if len(pred_list) == 0:
            continue
        #print("pred_list----------",pred_list)
        #print("rel_set---------------",rel_set)

        mrr=0.0
        hit_num = 0.0
        ndcg = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] == rel_set:
                mrr =1/(i+1)
                hit_num = 1
                ndcg = 1 / math.log(i + 2, 2)
                break

        hrs.append(hit_num)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        
    avg_hr = np.mean(hrs)*100
    avg_mrr = np.mean(mrrs)*100
    avg_ndcg = np.mean(ndcgs)*100
    
    
    print('HR={:.3f}  | MRR={:.3f} | NDCG={:.3f} |Invalid users={}'.format(
             avg_hr, avg_mrr, avg_ndcg,len(invalid_users)))
    return avg_hr, avg_mrr, avg_ndcg
    

     
 

#step==some values  
def batch_beam_search(session_rep,kz,env, model, session_dict, uids, sids,targets,batch_sessions, device, topk):

    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)
    state_pool = env.reset(session_dict, sids,targets, batch_sessions, uids)

    path_pool = env._batch_path  # list of list, size=bs

    probs_pool = [[] for _ in sids]



    for hop in range(len(topk)):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        leng1=[]

        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        action_embedding = env.get_action_embedding(path_pool, acts_pool)
        action_embedding = torch.FloatTensor(action_embedding).to(device)
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs,_ ,_= model((state_tensor, actmask_tensor),session_rep, action_embedding,kz)  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()


        new_path_pool, new_probs_pool = [], []

        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            x=0
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id, _ = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]

                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
                x=x+1
            leng1.append(x)
            
        path_pool = new_path_pool
        probs_pool = new_probs_pool



        if hop < len(topk)-1:
            state_pool = env._batch_get_state(path_pool)
        

        session_rep_new=[]
        for i in range(session_rep.shape[0]):
            for ii in range(leng1[i]):
                session_rep_new.append(session_rep[i,:].tolist())
        session_rep=np.array(session_rep_new)
        session_rep = torch.FloatTensor(session_rep).to(device)

    return path_pool, probs_pool







def evaluate_paths(args,results, session_dict, session_user_dict,  train_labels, test_labels):
    embeds = load_embed(args.dataset)
    product_embeds = embeds[PRODUCT]
    user_embeds = embeds[USER]  #user
    scores = np.matmul(user_embeds , product_embeds.T)

    # 1) 
    pred_paths = {sid: {} for sid in test_labels}
    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != PRODUCT:
            continue
        sid = path[0]
        if sid not in pred_paths:
            continue
        pid = path[-1][2]
        if pid not in pred_paths[sid]:
            pred_paths[sid][pid] = []

        path_score = np.dot(session_dict[sid],product_embeds[pid])
        path_prob = reduce(lambda x, y: x * y, probs)
        #print("path_prob",path_prob)
        pred_paths[sid][pid].append((path_score, path_prob, path))

    # 2) 
    best_pred_paths = {}
    for sid in pred_paths:
        uid=session_user_dict[sid]
        train_pids = set(train_labels[uid])
        best_pred_paths[sid] = []
        for pid in pred_paths[sid]:
            if pid in train_pids:
                continue
            sorted_path = sorted(pred_paths[sid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[sid].append(sorted_path[0])

    # 3) 
    sort_by = 'prob'
    pred_labels = {}
    best_hr = [0.0,0.0,0.0]
    best_mrr = [0.0,0.0,0.0]
    best_ndcg = [0.0,0.0,0.0]
    topkk=[5,10,20]
    for ik in range(3):
        k=topkk[ik]
        for sid in best_pred_paths:           
            if sort_by == 'score':
                sorted_path = sorted(best_pred_paths[sid], key=lambda x: (x[0], x[1]), reverse=True)
            elif sort_by == 'prob':
                sorted_path = sorted(best_pred_paths[sid], key=lambda x: (x[1], x[0]), reverse=True)
            top10_pids = [p[-1][2] for _, _, p in sorted_path[:k]] 
            #print(sorted_path)
            if args.add_products and len(top10_pids) < k:
                uid=session_user_dict[sid]
                train_pids = set(train_labels[uid])
                cand_pids = np.argsort(scores[uid])
                for cand_pid in cand_pids[::-1]:
                    if cand_pid in train_pids or cand_pid in top10_pids:
                        continue
                    top10_pids.append(cand_pid)
                    if len(top10_pids) >= k:
                        break
           
            pred_labels[sid] = top10_pids
        print("metrics@:",k)
        best_hr[ik], best_mrr[ik], best_ndcg[ik] = evaluate(pred_labels, test_labels,k)
        #print("pathhhhhhhhhhhh",best_hr[ik], best_mrr[ik], best_ndcg[ik])
    #print("sorted_path[:2]",sorted_path[:2])
    return best_hr, best_mrr, best_ndcg

def test(args,model,env,session,test_dataloader,session_user_dict,train_labels,test_labels):
    model.eval()
    session.eval()

    with torch.no_grad():
        all_paths, all_probs, all_targets = [], [], []
        test_dataloader.reset()
        session_dic = {}
        kz=0
        while test_dataloader.has_next():
            batch_sessions, seq_length, targets, batch_uids, batch_sids = test_dataloader.get_batch(kz)
            # print('targets',targets)

            if base_model=='srgnn' or base_model=='GASAN':
                sequences=batch_sessions.transpose(0,1)
                x = []
                batch=[]
                senders,receivers=[],[]
                i,j=0,0
                for sequence in sequences:
                    sender = []
                    nodes = {}
                    for node in sequence:
                        if node not in nodes:
                            nodes[node] = i
                            x.append([node])
                            i += 1
                        batch.append(j)
                        sender.append(nodes[node])
                    j += 1
                    receiver = sender[:]
                    if len(sender) != 1:
                        del sender[-1]
                        del receiver[0]
                    senders.extend(sender)
                    receivers.extend(receiver)
                edge_index = torch.tensor([senders, receivers], dtype=torch.long)
                x = torch.tensor(x, dtype=torch.long)
                batch = torch.tensor(batch, dtype=torch.long)
                session_rep=session(x,edge_index,batch)
            elif base_model=='narm':
                session_rep,score=session(batch_sessions.to(args.device),seq_length) #[bs,embed_dim]
            elif base_model=='gru4rec':
                session_rep=session(batch_sessions.to(args.device),seq_length) #[bs,embed_dim]
            elif base_model=='bert4rec':
                if batch_sessions.shape[0]<5:
                    batch_sessions=torch.cat((batch_sessions,torch.zeros(5-batch_sessions.shape[0],batch_sessions.shape[1],dtype=torch.long)))           
                session_rep=session(batch_sessions.transpose(0,1).to(args.device))

           
            dict1 = dict(zip(batch_sids, session_rep.tolist()))
            session_dic.update(dict1)
            batch_sessions=batch_sessions.transpose(0,1)
            paths, probs = batch_beam_search(session_rep,kz,env, model, session_dic, batch_uids, batch_sids, targets,batch_sessions,args.device,topk=args.topk)
            # print('reduce(lambda x, y: x * y, probs)',reduce(lambda x, y: x * y, probs))
            kz=1
            all_paths.extend(paths)
            all_probs.extend(probs)
            all_targets.extend(targets)

    predicts = {'paths': all_paths, 'probs': all_probs, 'targets': all_targets}
    best_hr, best_mrr, best_ndcg = evaluate_paths(args,predicts, session_dic, session_user_dict, train_labels, test_labels)
    return best_hr, best_mrr, best_ndcg




def get_loss(rewards,saved_actions,entropy_act, score,target,gamma,weight,criterion):
    device=torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    if len(rewards) <=0:
        del rewards[:]
        del saved_actions[:]
        del entropy_act[:]
        return 0.0, 0.0, 0.0
    batch_rewards = np.vstack(rewards).T  # numpy array of [bs, #steps]
    batch_rewards = torch.FloatTensor(batch_rewards).to(device)
    num_steps = batch_rewards.shape[1]
    for i in range(1, num_steps):
        batch_rewards[:, num_steps - i - 1] += gamma * batch_rewards[:, num_steps - i]
    actor_loss = 0
    entropy_act_loss = 0
    entropy_loss = 0
    
    for i in range(0, num_steps):
        log_prob_pai,value = saved_actions[i]
        #print(i,"\n",batch_rewards)
        advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
        actor_loss += -log_prob_pai * advantage.detach()  # Tensor of [bs, ]
       # log_prob = log_prob_pai[i] * batch_rewards[:, -1]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1] #user
        #log_prob = torch.tensor(batch_rewards.shape[0]*[1.]).to(device) * batch_rewards[:, -1]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1] #user
        #actor_loss += -log_prob  # Tensor of [bs, ]
        entropy_act_loss+= -entropy_act[i]

    actor_loss = actor_loss.mean()
    entropy_act_loss = entropy_act_loss.mean()
    entropy_loss = criterion(score, target)
    #loss = actor_loss+weight*entropy_act_loss+entropy_loss
    loss = 0.2*actor_loss+entropy_loss
    return loss, actor_loss, entropy_act_loss, entropy_loss

def main():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=150, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=100, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=2, help='Max path length.')  
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-2, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0.7, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=400, help='number of samples')
    parser.add_argument('--embed_dim', type=int, default=400, help='item embedding size of NARM') 
    parser.add_argument('--state_dim', type=int, default=400, help='dimension of state vector')
    parser.add_argument('--add_products', type=boolean, default=True, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[100,1], help='number of samples')  
    
#    parser.add_argument('--bert_max_len', type=int, default=5, help='Length of sequence for bert')
#    parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
#    parser.add_argument('--bert_num_heads', type=int, default=5, help='Number of heads for multi-attention')
#    parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
#    parser.add_argument('--bert_hidden_units', type=int, default=400, help='Size of hidden vectors (d_model)')
    
    args = parser.parse_args()

    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()

