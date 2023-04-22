#!/usr/bin/env python3

from .paths import *
from .common_helper import load_saved_data
from .visualization import render

import networkx as nx
from sklearn.neighbors import NearestNeighbors
from networkx.algorithms.shortest_paths import astar_path
import numpy as np
from tqdm import tqdm
import torch
import sys
sys.path.append(geometry_folder)

from core.geometry_utils import LERP


def riemannian_distance(H, start, end, num_steps):
    interp = LERP(start, end, num_steps)
    l = 0
    for i in range(len(interp) - 1):
        diff = interp[i + 1] - interp[i]
        l += np.sqrt(diff @ H @ diff.T)
    return l


def from_saved_data():
    saved_data_list = [
        f'{DEFAULT_SAVE_FOLDER}/encoder_sample_class2',
        f'{DEFAULT_SAVE_FOLDER}/encoder_sample_class7',
        f'{DEFAULT_SAVE_FOLDER}/calculate_actor_encoder_5_1404_0608',
        f'{DEFAULT_SAVE_FOLDER}/calculate_actor_encoder_6_1404_0708'
    ]

    all_feats = []
    all_hs = []
    all_name = []
    for filename in saved_data_list:
        _, _, feat, name, h = load_saved_data(filename)
        all_feats.append(feat)
        all_hs.append(hs)
        all_name.append(name)

    all_feats = np.concatenate(all_feats, axis=0)
    all_hs = np.concatenate(all_hs, axis=0)
    all_name = np.concatenate(all_name, axis=0)
    return all_feats, all_hs, all_name


def create_graph(wrapper, n_steps, n_neighbors, filename="paths_encoder"):
    all_feats, all_hs, all_name = from_saved_data()
    n_data = len(all_feats)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(all_feats)

    G = nx.Graph()
    for i in range(n_data):
        n_attr = {"name": all_name[i]}
        G.add_node(i, **n_attr)

    for i in tqdm(range(n_data)):
        distances, indices = knn.kneighbors(all_feats[i].reshape(1, -1))
        distances = distances[0]
        indices = indices[0]
        for ix, dist in zip(indices, distances):
            if (i, ix) in G.edges or (ix, i) in G.edges or i == ix:
                continue

            L_riemann = riemannian_distance(all_hs[i], all_feats[i], all_feats[ix:ix+1], n_steps)
            L_euclidean = dist
            edge_attr = {'weight': float(1/L_riemann),
                         'weight_euclidean': float(1/L_euclidean),
                         'distance_riemann': float(L_riemann),
                         'distance_euclidean': float(L_euclidean)}
            G.add_edge(i, ix, **edge_attr)

    path_feats = []
    path = astar_path(G, 0, 50)
    print(path)
    for i in path:
        path_feats.append(all_feats[i])
    save_name = f"{DEFAULT_SAVE_FOLDER}/{filename}.npz"
    np.savez(save_name, path=path_feats)
    return save_name


def generate_visualization(wrapper, start_code, end_code, num_step, save_folder, name="straight", start_ind=0):
    interp = LERP(start_code, end_code, num_step)
    for ni, feat in enumerate(interp):
        vid = wrapper.generate(torch.tensor(np.expand_dims(feat, axis=0), dtype=torch.float).to("cuda"))
        render(vid, wrapper.param, f"{name}_{start_ind + ni}.gif", save_folder)
