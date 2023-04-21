#!/usr/bin/env python3


import torch

import sys
from tools.paths import *

from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import os

from tools.actor_wrapper import ActorWrapper, make_actor_wrapper
from tools.calculate_sample_helper import measure_consistency, calculate_across_manifold
from tools.clip_wrapper import ClipWrapper, make_clip_wrapper, make_clip_scorer
from tools.generate_sample_helper import test_hessian
from tools.geodesic_helper import generate_visualization, create_graph
from tools.sim_score import FIDScore, EncoderScore, FIDSubsequence, EuclideanScore, \
    SkelEmbeddingScore, make_skel_scorer, LowerBodyScore
from tools.vae_operator import calculate_full_hessian


def make_wrapper(name):
    if name == "actor":
        wrapper = make_actor_wrapper(checkpoint_default_paths[name])
    elif name == "clip":
        wrapper = make_clip_wrapper(checkpoint_default_paths[name])
    else:
        print(f"No such wrapper {name}")
        exit(1)
    return wrapper


def make_scorer(name):
    if name == "fid":
        scorer = FIDScore()
    elif name == "subseq":
        scorer = FIDSubsequence()
    elif name == "encoder":
        scorer = make_clip_scorer(scorer_default_paths[name])
    elif name == "euc":
        scorer = EuclideanScore()
    elif name == "skel":
        scorer = make_skel_scorer(scorer_default_paths[name])
    elif name == "low":
        scorer = LowerBodyScore()
    else:
        print(f"No such scorer {name}")
        exit(1)
    return scorer


def anisotropy(savedir):
    import glob
    path = os.path.join(savedir, "*.npz")
    saved_files = glob.glob(path)
    for f in saved_files:
        print(os.path.basename(f))
        data = np.load(f)
        val = data["eva"]
        vec = data["evc"]
        H = data["H"]
        sortidx = np.argsort(-val)
        sval = val[sortidx]
        svec = vec[:, sortidx]
        val_sum = sum(sval)
        for i in range(len(sval)):
            # approx = svec[:, :i+1] @ np.diag(sval[:i+1]) @ svec[:, :i+1].T
            # diff = approx - H
            # cc = 1 - np.linalg.norm(diff, 'fro')
            # cc = np.corrcoef(H.flatten(), approx.flatten())[0, 1]
            partial_sum = sum(sval[:i+1])
            cc = partial_sum / val_sum
            if cc >= 0.99999:
                print(f"[0.99999] {i+1}: {cc}")
                break
            if cc >= 0.9999:
                print(f"[0.9999] {i+1}: {cc}")
                continue
            if cc >= 0.999:
                print(f"[0.999] {i+1}: {cc}")
                continue
            if cc >= 0.99:
                print(f"[0.99] {i+1}: {cc}")
                continue


def genfull(args):
    now = datetime.now()
    time_suffix = now.strftime("%d%m_%H%M")
    seed = 20
    torch.manual_seed(seed)

    folder = f"{DEFAULT_SAVE_FOLDER}/genfull_{time_suffix}"
    os.makedirs(folder, exist_ok=True)
    for w in available_wrappers:
        wrapper = make_wrapper(w)
        for s in available_dist_functions:
            scorer = make_scorer(s)
            feat = wrapper.sample_vector(sampn=1, sample_class=None)
            eva_f, evc_f, H_f = calculate_full_hessian(wrapper, feat, scorer)
            np.savez(os.path.join(folder, f"{w}_{s}.npz"), eva=eva_f, evc=evc_f, H=H_f, feat=feat.cpu().detach().numpy())
            print(f"{w}_{s}")


def samples_and_visualize(args):
    now = datetime.now()
    time_suffix = now.strftime("%d%m_%H%M")
    save_folder = f"{DEFAULT_SAVE_FOLDER}/visualize_{args.wrapper}_{args.scorer}_{args.maxdist}_{time_suffix}"
    wrapper = make_wrapper(args.wrapper)
    scorer = make_scorer(args.scorer)
    if args.sample_class is None:
        sc = None
    else:
        sc = int(args.sample_class)
    test_hessian(wrapper, scorer, save_folder,
                 eiglist=args.eiglist,
                 samples=args.num_samples,
                 cl=sc,
                 maxdist=args.maxdist,
                 cut_off=args.cutoff)
    print(f"Visualization saved to {save_folder}")


def sample_and_calculate(args):
    seed = 0
    torch.manual_seed(seed)
    now = datetime.now()
    time_suffix = now.strftime("%d%m_%H%M")
    wrapper = make_wrapper(args.wrapper)
    scorer = make_scorer(args.scorer)
    to_calculate = args.sample_class.split(',')
    for cl in to_calculate:
        if cl == "None":
            sample_cl = None
        else:
            sample_cl = int(cl)
        folder = f"{DEFAULT_SAVE_FOLDER}/calculate_{args.wrapper}_{args.scorer}_{cl}_{time_suffix}"
        calculate_across_manifold(wrapper, scorer, trials=args.num_trials, savedir=folder, class_name=sample_cl, cutoff=args.cutoff)
        measure_consistency(folder, f"{args.wrapper}-{args.scorer}")
    print(f"Calculations saved to {folder}")


def sample_full(args):
    scorers = args.slist.split(',')
    if scorers[0] == "all":
        scorers = available_dist_functions
    wrappers = args.wlist.split(',')
    if wrappers[0] == "all":
        wrappers = available_wrappers
    now = datetime.now()
    time_suffix = now.strftime("%d%m_%H%M")
    cutoff = {"actor": 20, "clip": 50}
    for w in wrappers:
        print(w)
        wrapper = make_wrapper(w)
        for s in scorers:
            print(s)
            scorer = make_scorer(s)
            folder = f"{DEFAULT_SAVE_FOLDER}/calculate_{w}_{s}_None_{time_suffix}"
            calculate_across_manifold(wrapper, scorer, trials=args.num_trials, savedir=folder, class_name=None, cutoff=cutoff[w])
            measure_consistency(folder, f"{w}-{s}")
            print(folder)


def geodesic_exp(args):
    seed = 0
    torch.manual_seed(seed)
    wrapper = make_wrapper(args.wrapper)

    if args.gen:
        filename = create_graph(wrapper, 0, 70, filename=f"geodesic-{args.wrapper}-{args.scorer}")
        print(filename)
        return

    now = datetime.now()
    time_suffix = now.strftime("%d%m_%H%M")
    path = np.load(args.path)
    feats = path["path"]
    print(len(feats))
    save_folder = f"{DEFAULT_SAVE_FOLDER}/geodesic_{args.wrapper}_{args.scorer}_{time_suffix}"
    os.makedirs(save_folder, exist_ok=True)

    # straight line
    if args.straight:
        start_code = feats[0]
        end_code = feats[-1]
        generate_visualization(wrapper, start_code, end_code, args.linear, save_folder)

    for i in range(len(feats) - 1):
        s = feats[i]
        e = feats[i + 1]
        generate_visualization(wrapper, s, e, args.step, save_folder,
                               name="riemann", start_ind=args.step * i)


if __name__ == '__main__':
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(help='commands', dest='task')
    visualize_parser = subparsers.add_parser('visualize')
    visualize_parser.add_argument("--num_samples", default=10, type=int)
    visualize_parser.add_argument("--sample_class", default=None)
    visualize_parser.add_argument("--maxdist", default=3.0, type=float)
    visualize_parser.add_argument("--eiglist", default="0,2,4,8,16")
    visualize_parser.add_argument("--cutoff", default=20, type=int)
    visualize_parser.add_argument("--wrapper", default="actor", choices=available_wrappers)
    visualize_parser.add_argument("--scorer", default="fid", choices=available_dist_functions)

    calculate_parser = subparsers.add_parser('calculate')
    calculate_parser.add_argument("--num_trials", default=50, type=int)
    calculate_parser.add_argument("--sample_class", default="None")
    calculate_parser.add_argument("--cutoff", default=20, type=int)
    calculate_parser.add_argument("--wrapper", default="actor", choices=available_wrappers)
    calculate_parser.add_argument("--scorer", default="fid", choices=available_dist_functions)

    anisotropy_parser = subparsers.add_parser('anisotropy')
    anisotropy_parser.add_argument("--dir", default=f"{DEFAULT_SAVE_FOLDER}/genfull_1904_0307")

    geodesic_parser = subparsers.add_parser('geodesic')
    geodesic_parser.add_argument("--gen", action='store_true')
    geodesic_parser.add_argument("--path", default=f"{DEFAULT_SAVE_FOLDER}/paths_encoder.npz")
    geodesic_parser.add_argument("--straight", action='store_true')
    geodesic_parser.add_argument("--linear", default=20, type=int)
    geodesic_parser.add_argument("--step", default=5, type=int)
    geodesic_parser.add_argument("--wrapper", default="actor", choices=available_wrappers)


    genfull_parser = subparsers.add_parser('genfull')

    samplefull_parser = subparsers.add_parser('samplefull')
    samplefull_parser.add_argument("--wlist", default="actor", help="List of wrappers, separated by comma")
    samplefull_parser.add_argument("--slist", default="fid", help="List of distance functions, separated by comma")
    samplefull_parser.add_argument("--num_trials", default=50, type=int)


    args = parser.parse_args()
    if args.task == "visualize":
        samples_and_visualize(args)
    elif args.task == "calculate":
        sample_and_calculate(args)
    elif args.task == "geodesic":
        geodesic_exp(args)
    elif args.task == "anisotropy":
        anisotropy(args.dir)
    elif args.task == "genfull":
        genfull(args)
    elif args.task == "samplefull":
        sample_full(args)
