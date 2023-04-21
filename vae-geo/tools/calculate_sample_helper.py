#!/usr/bin/env python3

from .paths import *
from .common_helper import load_saved_data
from .vae_operator import vae_hessian

from time import time
import sys
import os
from tqdm import tqdm
import numpy as np

sys.path.append(geometry_folder)
from core.hessian_analysis_tools import compute_hess_corr, compute_vector_hess_corr, \
    plot_consistentcy_mat, plot_consistency_hist, plot_consistency_example
from core import plot_spectra


def calculate_across_manifold(wrapper, dist, trials=30, savedir=DEFAULT_SAVE_FOLDER, class_name=None, cutoff=20):
    os.makedirs(savedir, exist_ok=True)
    cl = [class_name] * trials
    start = time()
    for i in tqdm(range(trials)):
        feat = wrapper.sample_vector(sampn=1, sample_class=cl[i])
        eva, evc, H = vae_hessian(wrapper, feat, dist, cut_off=cutoff)
        np.savez(os.path.join(savedir, f"Hessian_BP_{i}.npz"), eva=eva, evc=evc, H=H, feat=feat.cpu().detach().numpy())
    print(f"Took {time() - start}")


def measure_consistency(savedir, title_str, numsample=5):
    eva, evc, feat, name, _ = load_saved_data(savedir)
    plot_spectra(eva, savename=f"{title_str}-spectrum", figdir=savedir, titstr=title_str)
    print("Computing hessian correlation")
    corr_mat_log, corr_mat_lin = compute_hess_corr(eva, evc, figdir=savedir, use_cuda=False, savelabel=title_str)
    print("Computing hessian vector correlation")
    corr_mat_vec = compute_vector_hess_corr(eva, evc, figdir=savedir, use_cuda=False, savelabel=title_str)
    plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=savedir, titstr=title_str, savelabel=title_str)
    plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=savedir, titstr=title_str, savelabel=title_str)
    plot_consistency_example(eva, evc, figdir=savedir, nsamp=numsample, titstr=title_str, savelabel=title_str)
