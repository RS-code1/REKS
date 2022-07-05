from __future__ import absolute_import, division, print_function

import warnings

warnings.filterwarnings('ignore')
import os
import sys
import argparse
from math import log
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gzip
import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings('ignore')

from utils import *


# from data_utils import AmazonDataset


class KnowledgeGraph(object):

    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_reviews(dataset)
        self._load_knowledge(dataset)
        self._clean()
        self.top_matches = None

    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities():
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                self.G[entity][eid] = {r: [] for r in get_relations(
                    entity)}  ### entity=user-->self.G['user'][i]={PURCHASE: [],MENTION: []}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_reviews(self, dataset, word_tfidf_threshold=0.1, word_freq_threshold=5000):
        print('Load reviews...')
        num_edges = 0
        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data
            self._add_edge(USER, uid, PURCHASE, PRODUCT, pid)
            num_edges += 2
        print('Total {:d} user_product edges.'.format(num_edges))

    def _load_knowledge(self, dataset):

        for relation in [CO_OCCR]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(PRODUCT, relation)  
                    self._add_edge1(PRODUCT, pid, relation, et_type, eid)
                    num_edges += 1
            print('Total {:d} {:s} edges.'.format(num_edges, relation))
            
        for relation in [PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(PRODUCT, relation)  
                    self._add_edge(PRODUCT, pid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))
            
            

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)  
        #undirected---directed
        self.G[etype2][eid2][relation].append(eid1)
        
    def _add_edge1(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)  
        #self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

  
    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count
    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]



