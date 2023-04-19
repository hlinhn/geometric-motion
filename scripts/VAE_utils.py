import torch

import sys
sys.path.append('/home/linh/projects/GAN_Geometry')

from argparse import ArgumentParser
from datetime import datetime
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from core import hessian_compute
from core import plot_spectra
from core.GAN_hvp_operator import Operator
from core.geometry_utils import LExpMap, SExpMap, LERP
from core.hessian_analysis_tools import average_H
from core.lanczos_generalized import lanczos
from core.sim_score import FIDScore, EncoderScore, FIDSubsequence, EuclideanScore, SkelEmbeddingScore, make_skel_scorer, LowerBodyScore
from core.visualization import render, plot_3d
from core.actor_wrapper import ActorWrapper, make_actor_wrapper
from core.clip_wrapper import ClipWrapper, make_clip_wrapper, make_clip_scorer


class VAEHVPOperator(Operator):
    def __init__(self, model, z, criterion, use_gpu=True):
        if use_gpu:
            device = "cuda"
            self.device = device
        if hasattr(model, "parameters"):
            for param in model.parameters():
                param.requires_grad_(False)
        if hasattr(criterion, "parameters"):
            for param in criterion.parameters():
                param.requires_grad_(False)
        self.model = model
        self.criterion = criterion
        self.select_z(z)

    def select_z(self, z):
        self.z = z.detach().clone().float().to(self.device)
        self.size = self.z.numel()
        self.perturb_vec = 0.0001 * torch.randn((1, self.size), dtype=torch.float32).requires_grad_(True).to(self.device)
        self.ref = self.model.generate(self.z)
        perturbed = self.model.generate(self.z + self.perturb_vec)
        d_sim = self.criterion(self.ref, perturbed)
        gradient = torch.autograd.grad(d_sim, self.perturb_vec, create_graph=True, retain_graph=True)[0]
        self.gradient = gradient.view(-1)

    def select_z_size(self, z):
        self.select_z(z)
        self.size = self.perturb_vec.numel()

    def zero_grad(self):
        for p in [self.perturb_vec]:
            if p.grad is not None:
                p.grad.data.zero_()

    def apply(self, vec):
        self.zero_grad()
        grad_grad = torch.autograd.grad(self.gradient, self.perturb_vec, grad_outputs=vec, only_inputs=True, retain_graph=True)
        vec_prod = grad_grad[0].view(-1)
        return vec_prod

    def vHv_form(self, vec):
        vec_prod = self.apply(vec)
        vhv = (vec_prod * vec).sum()
        return vhv


def vae_hessian(G, feat, dist, device="cuda", cut_off=None):
    metrichvp = VAEHVPOperator(G, feat, dist)
    if not cut_off:
        cut_off = feat.numel() // 2 - 1
    eigvals, eigvects = lanczos(metrichvp, num_eigenthings=cut_off, use_gpu=True)
    # print(eigvals)
    eigvects = eigvects.T
    H = eigvects @ np.diag(eigvals) @ eigvects.T
    # print(f"Took {time() - start}")
    return eigvals, eigvects, H


def get_full_hessian(l, param):
    # print(l)
    hs = param.numel()
    hessian = torch.zeros(hs, hs)
    loss_grad = torch.autograd.grad(l, param, create_graph=True, retain_graph=True, only_inputs=True)[0].view(-1)
    # print(loss_grad)
    for idx in range(hs):
        grad2rd = torch.autograd.grad(loss_grad[idx], param, create_graph=False, retain_graph=True, only_inputs=True)
        hessian[idx] = grad2rd[0].view(-1)
    return hessian.cpu().data.numpy()


def calculate_full_hessian(G, feat, dist, device="cuda"):
    start = time()
    ref_vec = feat.detach().clone().float().to(device)
    mov = ref_vec.float().detach().clone().requires_grad_(True)
    # perb = 0.0001 * torch.randn((1, feat.numel()), dtype=torch.float32).requires_grad_(True).to(device)
    mov = mov # + perb
    v1 = G.generate(ref_vec)
    v2 = G.generate(mov)
    d = dist(v1, v2)
    H = get_full_hessian(d, mov)
    eigval, eigvec = np.linalg.eigh(H)
    # print(eigval)
    print(f"Took {time() - start}")
    return eigval, eigvec, H


def visualize_action(eigvec, ref_codes, G, maxdist=3.0, rown=7, name="", save_folder="/home/linh/vae-geo"):
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
                       maxdist=3.0, namestr="test", figdir="/home/linh/vae-geo"):
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
    plt.savefig(os.path.join(figdir, f"{namestr}-imdistcrv.jpg"))


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


def compare_full_vs_estimate(args):
    wrapper = make_wrapper(args.wrapper)
    dist = make_scorer(args.scorer)
    save_data = {}
    saved_folder = f"/home/linh/vae-geo/compare_{args.wrapper}_{args.scorer}_{args.cutoff}"
    os.makedirs(saved_folder, exist_ok=True)
    for i in range(12):
        feat = wrapper.sample_vector(sampn=1, sample_class=i)
        eva_BP, evc_BP, H_BP = vae_hessian(wrapper, feat, dist, cut_off=args.cutoff)
        save_data[f"estimate_{i}"] = (eva_BP, evc_BP, H_BP)
        plt.plot(eva_BP[::-1])
        plt.title("Top Hessian spectrum")
        plt.ylabel("Eigenvalue")
        plt.xlabel("Rank")
        plt.savefig(f"{saved_folder}/spectrum_at_{i}.png")
        plot_spectra([eva_BP], titstr="Top spectrum", figdir=saved_folder, savename=f"spectrum_and_log_at_{i}")
        eva_f, evc_f, H_f = calculate_full_hessian(wrapper, feat, dist)
        save_data[f"full_{i}"] = (eva_f, evc_f, H_f)
        plot_spectra([eva_f], titstr="Top spectrum full", figdir=saved_folder, savename=f"full_spectrum_and_log_at_{i}")
        cc = np.corrcoef(H_f.flatten(), H_BP.flatten())[0][1]
        print(f"Correlation similarity of full and rank {args.cutoff} approximation at {i} is {cc}")

    with open(f'{saved_folder}/hessian_estimate.pkl', 'wb') as fp:
        pickle.dump(save_data, fp)


def calculate_across_manifold(wrapper, dist, trials=30, savedir="/home/linh/vae-geo", class_name=None, cutoff=20):
    os.makedirs(savedir, exist_ok=True)
    cl = [class_name] * trials
    start = time()
    for i in tqdm(range(trials)):
        feat = wrapper.sample_vector(sampn=1, sample_class=cl[i])
        eva, evc, H = vae_hessian(wrapper, feat, dist, cut_off=cutoff)
        np.savez(os.path.join(savedir, f"Hessian_BP_{i}.npz"), eva=eva, evc=evc, H=H, feat=feat.cpu().detach().numpy())
    print(f"Took {time() - start}")


def load_saved_data(savedir):
    import glob
    vals = []
    vecs = []
    feats = []
    name = []
    hs = []
    path = os.path.join(savedir, "Hessian_BP_*.npz")
    saved_files = glob.glob(path)
    cnt = 0
    for f in saved_files:
        # print(f)
        if cnt >= 50:
            break
        cnt += 1
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


def measure_consistency(savedir, title_str):
    from core.hessian_analysis_tools import compute_hess_corr, compute_vector_hess_corr, plot_consistentcy_mat, plot_consistency_hist, plot_consistency_example
    eva, evc, feat, name, _ = load_saved_data(savedir)
    plot_spectra(eva, savename=f"{title_str}-spectrum", figdir=savedir, titstr=title_str)
    # average = average_H(eva, evc)
    print("Computing hessian correlation")
    corr_mat_log, corr_mat_lin = compute_hess_corr(eva, evc, figdir=savedir, use_cuda=False, savelabel=title_str)
    print("Computing hessian vector correlation")
    corr_mat_vec = compute_vector_hess_corr(eva, evc, figdir=savedir, use_cuda=False, savelabel=title_str)
    fig1, fig2 = plot_consistentcy_mat(corr_mat_log, corr_mat_lin, figdir=savedir, titstr=title_str, savelabel=title_str)
    fig11, fig22 = plot_consistency_hist(corr_mat_log, corr_mat_lin, figdir=savedir, titstr=title_str, savelabel=title_str)
    fig3 = plot_consistency_example(eva, evc, figdir=savedir, nsamp=5, titstr=title_str, savelabel=title_str)


def make_wrapper(name):
    checkpoint_default_paths = {
        "actor": "/home/linh/projects/ACTOR/pretrained_models/humanact12/checkpoint_5000.pth.tar",
        "clip": "/home/linh/Downloads/motionclip/paper-model/checkpoint_0100.pth.tar"
    }
    if name == "actor":
        wrapper = make_actor_wrapper(checkpoint_default_paths[name])
    elif name == "clip":
        wrapper = make_clip_wrapper(checkpoint_default_paths[name])
    else:
        print(f"No such wrapper {name}")
        exit(1)
    return wrapper


def make_scorer(name):
    checkpoint_default_paths = {
        "encoder": "/home/linh/Downloads/motionclip/paper-model/checkpoint_0100.pth.tar",
        "skel": "/home/linh/projects/ACTOR/pretrained_models/humanact12/checkpoint_5000.pth.tar"
    }
    if name == "fid":
        scorer = FIDScore()
    elif name == "subseq":
        scorer = FIDSubsequence()
    elif name == "encoder":
        scorer = make_clip_scorer(checkpoint_default_paths[name])
    elif name == "euc":
        scorer = EuclideanScore()
    elif name == "skel":
        scorer = make_skel_scorer(checkpoint_default_paths[name])
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

    folder = f"/home/linh/vae-geo/genfull_{time_suffix}"
    os.makedirs(folder, exist_ok=True)
    for w in ["actor", "clip"]:
        wrapper = make_wrapper(w)
        for s in ["fid", "subseq", "encoder", "euc", "skel", "low"]:
            scorer = make_scorer(s)
            feat = wrapper.sample_vector(sampn=1, sample_class=None)
            eva_f, evc_f, H_f = calculate_full_hessian(wrapper, feat, scorer)
            np.savez(os.path.join(folder, f"{w}_{s}.npz"), eva=eva_f, evc=evc_f, H=H_f, feat=feat.cpu().detach().numpy())
            print(f"{w}_{s}")


def samples_and_visualize(args):
    now = datetime.now()
    time_suffix = now.strftime("%d%m_%H%M")
    save_folder = f"/home/linh/vae-geo/visualize_{args.wrapper}_{args.scorer}_{args.maxdist}_{time_suffix}"
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
    to_calculate = args.calculate_class.split(',')
    for cl in to_calculate:
        if cl == "None":
            sample_cl = None
        else:
            sample_cl = int(cl)
        folder = f"/home/linh/vae-geo/calculate_{args.wrapper}_{args.scorer}_{cl}_{time_suffix}"
        wrapper = make_wrapper(args.wrapper)
        scorer = make_scorer(args.scorer)
        calculate_across_manifold(wrapper, scorer, trials=args.num_trials, savedir=folder, class_name=sample_cl, cutoff=args.cutoff)
        measure_consistency(folder, f"{args.wrapper}-{args.scorer}")
    print(f"Calculations saved to {folder}")
    # compare_full_vs_estimate(actor)
    # output = test_generator(actor)


def sample_full(args):
    scorers = ["fid", "skel", "low"]
    now = datetime.now()
    time_suffix = now.strftime("%d%m_%H%M")
    cutoff = {"actor": 20, "clip": 50}
    for w in ["clip"]:
        print(w)
        wrapper = make_wrapper(w)
        for s in scorers:
            print(s)
            scorer = make_scorer(s)
            folder = f"/home/linh/vae-geo/calculate_{w}_{s}_None_{time_suffix}"
            calculate_across_manifold(wrapper, scorer, trials=args.num_trials, savedir=folder, class_name=None, cutoff=cutoff[w])
            measure_consistency(folder, f"{w}-{s}")
            print(folder)


def geodesic_exp(args):
    seed = 0
    torch.manual_seed(seed)
    now = datetime.now()
    time_suffix = now.strftime("%d%m_%H%M")
    wrapper = make_wrapper(args.wrapper)
    # create_graph(wrapper, 0, 70)
    # return
    path = np.load("/home/linh/vae-geo/paths_encoder.npz")
    feats = path["path"]
    print(len(feats))
    save_folder = f"/home/linh/vae-geo/geodesic_{args.wrapper}_{args.scorer}_{time_suffix}"
    os.makedirs(save_folder, exist_ok=True)
    # straight line
    # start_code = feats[0]
    # end_code = feats[-1]
    # interp = LERP(start_code, end_code, 40)
    # for ni, feat in enumerate(interp):
    #     vid = wrapper.generate(torch.tensor(np.expand_dims(feat, axis=0), dtype=torch.float).to("cuda"))
    #     render(vid, wrapper.param, f"straight_{ni}.gif", save_folder)

    for i in range(len(feats) - 1):
        s = feats[i]
        e = feats[i + 1]
        inte = LERP(s, e, 4)
        for ni, feat in enumerate(inte):
            vid = wrapper.generate(torch.tensor(np.expand_dims(feat, axis=0), dtype=torch.float).to("cuda"))
            render(vid, wrapper.param, f"riemann_{i * 10 + ni}.gif", save_folder)


def riemannian_distance(H, start, end):
    diff = start - end
    l = np.sqrt(diff @ H @ diff.T)
    return l


def create_graph(wrapper, n_steps, n_neighbors):
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors
    from networkx.algorithms.shortest_paths import astar_path

    _, _, feat1, name1, hs1 = load_saved_data('/home/linh/vae-geo/encoder_sample_class2')
    _, _, feat2, name2, hs2 = load_saved_data('/home/linh/vae-geo/encoder_sample_class7')
    _, _, feat3, name3, hs3 = load_saved_data('/home/linh/vae-geo/calculate_actor_encoder_5_1404_0608')
    _, _, feat4, name4, hs4 = load_saved_data('/home/linh/vae-geo/calculate_actor_encoder_6_1404_0708')
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
    np.savez("/home/linh/vae-geo/paths_encoder.npz", path=path_feats)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--wrapper", default="actor", choices=["actor", "clip"])
    parser.add_argument("--scorer", default="fid", choices=["fid", "subseq", "encoder", "euc", "skel", "low"])
    parser.add_argument("--task", default="visualize", choices=["visualize", "calculate", "geodesic", "compare", "anisotropy", "genfull", "samplefull"])

    parser.add_argument("--cutoff", default=20, type=int)
    parser.add_argument("--eiglist", default="0,2,4,8,16")
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--sample_class", default=None)
    parser.add_argument("--maxdist", default=3.0, type=float)

    parser.add_argument("--num_trials", default=50, type=int)
    parser.add_argument("--calculate_class", default="None")

    parser.add_argument("--dir", default="/home/linh/vae-geo/genfull_1904_0307")

    args = parser.parse_args()

    if args.task == "visualize":
        samples_and_visualize(args)
    elif args.task == "calculate":
        sample_and_calculate(args)
    elif args.task == "geodesic":
        geodesic_exp(args)
    elif args.task == "compare":
        compare_full_vs_estimate(args)
    elif args.task == "anisotropy":
        anisotropy(args.dir)
    elif args.task == "genfull":
        genfull(args)
    elif args.task == "samplefull":
        sample_full(args)
