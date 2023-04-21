#!/usr/bin/env python3


from .paths import *
from .visualization import render
from .vae_operator import vae_hessian

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(geometry_folder)

from core.geometry_utils import LExpMap, SExpMap, LERP


def visualize_action(eigvec, ref_codes, G, maxdist=3.0, rown=7, name="", save_folder=DEFAULT_SAVE_FOLDER):
    reflist = list(ref_codes)
    for idx, ref_code in enumerate(reflist):
        ref_code_send = ref_code.clone().detach().cpu().numpy()
        interp = LExpMap(ref_code_send, eigvec, rown, (-maxdist, maxdist))
        for ni, feat in enumerate(interp):
            vid = G.generate(torch.tensor(np.expand_dims(feat, axis=0), dtype=torch.float).to("cuda"))
            # plot_3d(vid, name=f"{name}_vid_{ni}.gif", save_folder=save_folder)
            render(vid, G.param, f"{name}_vid_{ni}.gif", save_folder)


def visualize_distance(ref_code, eigvect, eigval, G, dist,
                       eiglist=[0], distrown=19, rown=3,
                       maxdist=3.0, namestr="test", figdir=DEFAULT_SAVE_FOLDER):
    os.makedirs(figdir, exist_ok=True)
    refdata = G.generate(ref_code)
    ticks = np.linspace(-maxdist, maxdist, distrown, endpoint=True)
    visticks = np.linspace(-maxdist, maxdist, rown, endpoint=True)
    distmat = np.zeros((len(eiglist), distrown))
    ref_code_send = ref_code.clone().detach().cpu().numpy()
    for idx, eigi in enumerate(eiglist):
        interp = LExpMap(ref_code_send, eigvect[:, -eigi - 1], distrown, (-maxdist, maxdist))
        for ni, feat in enumerate(interp):
            feat_tens = torch.tensor(np.expand_dims(feat, axis=0), dtype=torch.float).to("cuda")
            vid = G.generate(feat_tens)
            with torch.no_grad():
                distmat[idx][ni] = dist(refdata, vid)
    fig = plt.figure(figsize=[5, 5])
    for idx, eigi in enumerate(eiglist):
        plt.plot(ticks, distmat[idx, :], label="eig%d %.E" % (eigi + 1, eigval[-eigi - 1]), lw=2.5, alpha=0.7)
    plt.xticks(visticks)
    plt.ylabel("distance")
    plt.xlabel("L2 in latent space")
    plt.legend()
    plt.subplots_adjust(left=0.14, bottom=0.14)
    plt.savefig(os.path.join(figdir, f"imdist-{namestr}.jpg"))


def test_hessian(wrapper, dist, save_folder, eiglist, cl=0, samples=10, maxdist=3.0, cut_off=20):
    eig_vector_list = [int(x) for x in eiglist.split(',')]
    for i in tqdm(range(samples)):
        feat = wrapper.sample_vector(sampn=1, sample_class=cl)
        eva_BP, evc_BP, H_BP = vae_hessian(wrapper, feat, dist, cut_off=cut_off)
        visualize_distance(feat, evc_BP, eva_BP, wrapper, dist,
                           eiglist = eig_vector_list,
                           maxdist=maxdist,
                           namestr=f"class_{cl}_sample_{i}",
                           figdir=save_folder)
        for j in eig_vector_list:
            visualize_action(evc_BP.T[-j - 1], feat, wrapper,
                             name=f"class_{cl}_sample_{i}_eig_{j}",
                             save_folder=save_folder, maxdist=maxdist)
