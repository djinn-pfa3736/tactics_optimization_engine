import sys

import cv2
from PIL import Image
import imagehash

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from sklearn.cluster import OPTICS
from scipy.cluster.hierarchy import linkage, dendrogram

import pickle

import pdb

class Optimizer:

    def __init__(self):

        with open('spatial_temporal_grid.pickle', "rb") as f:
            self.spatial_temporal_grid = pickle.load(f)
        self.spatial_temporal_grid = self.spatial_temporal_grid/(np.sum(self.spatial_temporal_grid))
        _, _, interval = self.spatial_temporal_grid.shape

        self.dist_mat = np.zeros((interval, interval))

    def prepare(self, calc_dist_flag, max_eps, min_samples, method):

        if calc_dist_flag == 1:

            for i in range(interval - 1):
                for j in range(i + 1, interval):
                    print('*** (%s, %s) ***' % (i, j))
                    self.dist_mat[i, j] = self.calc_dist(self.spatial_temporal_grid[:,:,i], self.spatial_temporal_grid[:,:,j], i, j, interval, method)

            self.dist_mat = self.dist_mat + self.dist_mat.T
            with open('dist_mat.pickle', "wb") as f:
                pickle.dump(self.dist_mat, f)

        else:
            with open('dist_mat.pickle', "rb") as f:
                self.dist_mat = pickle.load(f)

        self.clust_res = OPTICS(metric='precomputed').fit(self.dist_mat)

    def calc_dist(self, dist1, dist2, t_idx1, t_idx2, period, method='kl'):

        rows, cols = dist1.shape
        delta_t = np.abs(t_idx2 - t_idx1)

        if delta_t < period/2:
            p = dist2
            q = dist1
        else:
            p = dist1
            q = dist2

        if method == 'kl':
            p = p.flatten()
            q = q.flatten()
            q += 1/(rows*cols)

            # result = np.sum(np.where(1e-8 < p, p*np.log(p/q), 0))
            result = entropy(p, qk=q, base=2)
        elif method == 'js':
            p = p.flatten()
            q = q.flatten()
            q += 1/(rows*cols)

            result = jensenshannon(p, q, base=2)
        elif method == 'l1':
            result = np.sum(np.abs(p - q))
        elif method == 'l2':
            result = np.sum((p - q)**2)
        elif method == 'wass':
            p_sig = np.array([np.array([p[i, j], i, j]) for i in range(rows) for j in range(cols)])
            q_sig = np.array([np.array([q[i, j], i, j]) for i in range(rows) for j in range(cols)])

            result = cv2.EMD(p_sig, q_sig, cv2.DIST_L2)
        elif method == 'bhat':
            result = -np.log(np.sum(np.sqrt(p*q)))
        elif method == 'phash':
            max_p = np.max(p)
            max_q = np.max(q)
            max_val = np.max([max_p, max_q])

            p_image = Image.fromarray(np.uint8(p/max_val*255))
            p_hash = imagehash.average_hash(p_image)
            q_image = Image.fromarray(np.uint8(q/max_val*255))
            q_hash = imagehash.average_hash(q_image)
            result = p_hash - q_hash

            # pdb.set_trace()
        else:
            print('Error: invalid method name is given.')
            exit(1)
        if result < 0:
            pdb.set_trace()

        return result
