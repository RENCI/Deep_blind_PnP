import os
import numpy as np
import cv2
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from config import get_config
from lib.data_loaders import make_data_loader
from lib.utils import load_model
from lib.timer import *
from lib.transformations import quaternion_from_matrix


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

logging.basicConfig(level=logging.INFO, format="")


def main(config):
    # loading the test data
    inf_data_loader = make_data_loader(configs, "infer", 1, num_threads=configs.val_num_thread, shuffle=False)

    # no gradients
    with torch.no_grad():
        # Model initialization
        Model = load_model(config.model)
        model = Model(config)

        # limited GPU
        if config.gpu_inds > -1:
            torch.cuda.set_device(config.gpu_inds)
            device = torch.device('cuda', config.gpu_inds)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)

        # load the weights
        if config.weights:
            checkpoint = torch.load(config.weights)
            model.load_state_dict(checkpoint['state_dict'])

        logging.info(model)

        # evaluation model
        model.eval()

        num_data = 0
        data_timer, matching_timer = Timer(), Timer()
        tot_num_data = len(inf_data_loader.dataset)

        data_loader_iter = inf_data_loader.__iter__()

        # for gpu memory consideration
        max_nb_points = 20000

        for batch_idx in range(tot_num_data):

            data_timer.tic()
            input_3d_xyz, input_2d_xy = data_loader_iter.next()
            data_timer.toc()

            p3d_cur = input_3d_xyz
            p2d_cur = input_2d_xy

            p2d_cur, p3d_cur = p2d_cur.to(device), p3d_cur.to(device)

            # Compute output
            matching_timer.tic()
            prob_matrix = model(p3d_cur, p2d_cur)
            matching_timer.toc()

            # compute the topK correspondences
            # note cv2.solvePnPRansac is not stable, sometimes wrong!
            # please use https://github.com/k88joshi/posest, and cite the original paper
            k = min(2000, round(p3d_cur.size(1) * p2d_cur.size(1)))  # Choose at most 2000 points in the testing stage
            _, P_topk_i = torch.topk(prob_matrix.flatten(start_dim=-2), k=k, dim=-1, largest=True, sorted=True)
            p3d_indices = P_topk_i / prob_matrix.size(-1)  # bxk (integer division)
            p2d_indices = P_topk_i % prob_matrix.size(-1)  # bxk

            torch.cuda.empty_cache()

            data_timer.reset()

            num_data += 1


if __name__ == '__main__':

    configs = get_config()

    configs.gpu_inds = 1
    # dataset dir
    # configs.data_dir = "/media/liu/data"
    # the used dataset
    configs.dataset = "megaDepth"
    # configs.dataset = "modelnet40"
    # pre-trained networks
    configs.debug_nb = "preTrained"

    configs.weights = os.path.join(configs.out_dir, configs.dataset, configs.debug_nb) + '/best_val_checkpoint.pth'

    main(configs)
