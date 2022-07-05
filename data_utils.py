from __future__ import absolute_import, division, print_function

import os
import numpy as np
import gzip
import pickle
from easydict import EasyDict as edict
import random
import warnings
warnings.filterwarnings('ignore')


class AmazonDataset(object):
    """This class is used to load data files and save in the instance."""

    def __init__(self, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir += '/'
        self.review_file = set_name + '.txt'
        self.load_entities()
        self.load_product_relations()
        self.load_reviews()

    def _load_file(self, filename):
        ###save columns name
        file=open(self.data_dir + filename)
        return [line.replace('\n','').strip() for line in file][1:]

    def load_entities(self):
        """Load 6 global entities from data files:
        `user`, `product`, `word`, `related_product`, `brand`, `category`.
        Create a member variable for each entity associated with attributes:
        - `vocab`: a list of string indicating entity values.
        - `vocab_size`: vocabulary size.
        """
        entity_files = edict(
                user='users.txt',
                product='product.txt',
                related_product='related_product.txt',
                brand='brands.txt',
                category='categories.txt',
        )

        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))   
            print('Load', name, 'of size', len(vocab))

    def load_reviews(self):
        """Load user-product reviews from train/test data files.
        Create member variable `review` associated with following attributes:
        - `data`: list of tuples (user_idx, product_idx, [word_idx...]).
        - `size`: number of reviews.
        - `product_distrib`: product vocab frequency among all eviews.
        - `product_uniform_distrib`: product vocab frequency (all 1's)
        - `word_distrib`: word vocab frequency among all reviews.
        - `word_count`: number of words (including duplicates).
        - `review_distrib`: always 1.
        """
        review_data = []  # (user_idx, product_idx, [word1_idx,...,wordn_idx])
        product_distrib = np.zeros(self.product.vocab_size)
        word_count = 0
        for line in self._load_file(self.review_file):
            try:
                arr = line.split('\t')
                user_idx = int(arr[0])
                product_idx = int(arr[1])
                word_indices = [int(i) for i in arr[2].split(' ')]  # list of word idx
                review_data.append((user_idx, product_idx, word_indices))
                product_distrib[product_idx] += 1
            except:
                print(line)
        self.review = edict(
                data=review_data,
                size=len(review_data),
                product_distrib=product_distrib,
                product_uniform_distrib=np.ones(self.product.vocab_size),
                word_count=word_count,
                review_distrib=np.ones(len(review_data)) #set to 1 now
        )
        print('Load review of size', self.review.size, 'word count=', word_count)

    def load_product_relations(self):
        """Load 5 product -> ? relations:
        - `produced_by`: product -> brand,
        - `belongs_to`: product -> category,
        - `also_bought`: product -> related_product,
        - `also_viewed`: product -> related_product,
        - `bought_together`: product -> related_product,
        - `co_occr`: product -> product,
        Create member variable for each relation associated with following attributes:
        - `data`: list of list of entity_tail indices (can be empty).
        - `et_vocab`: vocabulary of entity_tail (copy from entity vocab).
        - `et_distrib`: frequency of entity_tail vocab.
        """
        product_relations = edict(
                produced_by=('brand_p_b.txt', self.brand),  # (filename, entity_tail)
                belongs_to=('category_p_c.txt', self.category), 
                also_bought=('also_bought_p_p.txt', self.related_product),
                also_viewed=('also_viewed_p_p.txt', self.related_product),
                bought_together=('bought_together_p_p.txt', self.related_product),
                co_occr=('co_occr_p_p.txt',self.product)
        )
        for name in product_relations:

            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict(
                    data=[],
                    et_vocab=product_relations[name][1].vocab, #copy of brand, catgory ... 's vocab
                    et_distrib=np.zeros(product_relations[name][1].vocab_size) #[1] means self.brand ..
            )
            for line in self._load_file(product_relations[name][0]): 
                knowledge = []
                for x in line.split(' '):  # some lines may be empty
                    if len(x) > 0 and x!='""':
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                relation.data.append(knowledge)  
            setattr(self, name, relation)  ## self.produced_by/self.belongs_to
            print('Load', name, 'of size', len(relation.data))


    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.review.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.review.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min((np.sqrt(float(self.review.word_distrib[i]) / threshold) + 1) * threshold / float(self.review.word_distrib[i]), 1.0)


class AmazonDataLoader(object):
    """This class acts as the dataloader for training knowledge graph embeddings."""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.nums=len(dataset)
        self.reset()

    def reset(self):
        # Shuffle reviews order
        self._rand_perm = np.random.permutation(self.nums)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.nums)
        #print("numsssssss",self.nums)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch=self.dataset.iloc[batch_idx,:]
        self._has_next = self._has_next and end_idx < self.nums
        self._start_idx = end_idx
        return np.array(batch)


