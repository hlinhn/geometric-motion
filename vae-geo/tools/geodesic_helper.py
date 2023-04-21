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

def riemannian_distance(H, start, end):
    diff = start - end
    l = np.sqrt(diff @ H @ diff.T)
    return l


def create_graph(wrapper, n_steps, n_neighbors, filename="paths_encoder"):
    _, _, feat1, name1, hs1 = load_saved_data(f'{DEFAULT_SAVE_FOLDER}/encoder_sample_class2')
    _, _, feat2, name2, hs2 = load_saved_data(f'{DEFAULT_SAVE_FOLDER}/encoder_sample_class7')
    _, _, feat3, name3, hs3 = load_saved_data(f'{DEFAULT_SAVE_FOLDER}/calculate_actor_encoder_5_1404_0608')
    _, _, feat4, name4, hs4 = load_saved_data(f'{DEFAULT_SAVE_FOLDER}/calculate_actor_encoder_6_1404_0708')
    all_feats = np.concatenate((feat1, feat2, feat3, feat4), axis=0)
    all_hs = np.concatenate((hs1, hs2, hs3, hs4), axis=0)
    all_name = np.concatenate((name1, name2, name3, name4), axis=0)
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

            L_riemann = riemannian_distance(all_hs[i], all_feats[i], all_feats[ix:ix+1])
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
