#!/usr/bin/env python3


import glob
import os
import numpy as np


def load_saved_data(savedir, limit=None, filename="Hessian_BP_*.npz"):
    vals = []
    vecs = []
    feats = []
    name = []
    hs = []
    path = os.path.join(savedir, filename)
    saved_files = glob.glob(path)
    if limit is None:
        limit = len(saved_files)
    for idx, f in enumerate(saved_files):
        if idx >= limit:
            break
        data = np.load(f)
        val = data["eva"]
        vals.append(val)
        vec = data["evc"]
        vecs.append(vec)
        feat = data["feat"]
        feats.append(feat)
        H = data["H"]
        hs.append(H)
        name.append(f)
    vals = np.array(vals)
    feats = np.array(tuple(feats)).squeeze()
    return vals, vecs, feats, name, hs
