from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd

from utils import *
from data_utils import AmazonDataset, AmazonDataLoader
from transe_model import KnowledgeEmbedding


logger = None


def train(args):
    dataset_trans = pd.read_csv('./data/transe_train_data.csv')

    dataloader = AmazonDataLoader(dataset_trans, args.batch_size)
    dataset = load_dataset(args.dataset)

    model = KnowledgeEmbedding(dataset, args).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    steps = 0
    smooth_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Get training batch.
            batch_idxs = dataloader.get_batch()
            batch_idxs = torch.from_numpy(batch_idxs).to(args.device)

            # Train model.
            optimizer.zero_grad()
            train_loss = model(batch_idxs)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            smooth_loss += train_loss.item() / args.steps_per_checkpoint

            steps += 1
            if steps % args.steps_per_checkpoint == 0:
                logger.info('Epoch: {:02d} | '.format(epoch) +
                            'Smooth loss: {:.5f}'.format(smooth_loss))
                smooth_loss = 0.0
        torch.save(model.state_dict(), '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, epoch))
        extract_embeddings(args,epoch)

def extract_embeddings(args,epoch):
    """Note that last entity embedding is of size [vocab_size+1, d]."""
    model_file = '{}/transe_model_sd_epoch_{}.ckpt'.format(args.log_dir, epoch)
    print('Load embeddings', model_file)
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
    embeds = {
        USER: state_dict['user.weight'].cpu().data.numpy()[:-1],  # Must remove last dummy 'user' with 0 embed.
        PRODUCT: state_dict['product.weight'].cpu().data.numpy()[:-1],
        BRAND: state_dict['brand.weight'].cpu().data.numpy()[:-1],
        CATEGORY: state_dict['category.weight'].cpu().data.numpy()[:-1],
        RPRODUCT: state_dict['related_product.weight'].cpu().data.numpy()[:-1],

        PURCHASE: (
            state_dict['purchase'].cpu().data.numpy()[0],
            state_dict['purchase_bias.weight'].cpu().data.numpy()
        ),
        PRODUCED_BY: (
            state_dict['produced_by'].cpu().data.numpy()[0],
            state_dict['produced_by_bias.weight'].cpu().data.numpy()
        ),
        BELONG_TO: (
            state_dict['belongs_to'].cpu().data.numpy()[0],
            state_dict['belongs_to_bias.weight'].cpu().data.numpy()
        ),
        ALSO_BOUGHT: (
            state_dict['also_bought'].cpu().data.numpy()[0],
            state_dict['also_bought_bias.weight'].cpu().data.numpy()
        ),
        ALSO_VIEWED: (
            state_dict['also_viewed'].cpu().data.numpy()[0],
            state_dict['also_viewed_bias.weight'].cpu().data.numpy()
        ),
        BOUGHT_TOGETHER: (
            state_dict['bought_together'].cpu().data.numpy()[0],
            state_dict['bought_together_bias.weight'].cpu().data.numpy()
        ),
        CO_OCCR: (
            state_dict['co_occr'].cpu().data.numpy()[0],
            state_dict['co_occr_bias.weight'].cpu().data.numpy())
    }
    save_embed(args.dataset, embeds,epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY)
    parser.add_argument('--name', type=str, default='train_transe_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train.')  ##30
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for adam.')
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 lambda')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Clipping gradient.')
    parser.add_argument('--embed_size', type=int, default=400, help='knowledge embedding size.')
    parser.add_argument('--num_neg_samples', type=int, default=5, help='number of negative samples.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=200, help='Number of steps for checkpoint.')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    train(args)
  #  extract_embeddings(args)


if __name__ == '__main__':
    main()

