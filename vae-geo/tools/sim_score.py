#!/usr/bin/env python3


from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

from .paths import *
import sys
sys.path.append(actor_folder)
from src.evaluate.action2motion.models import load_classifier_for_fid
from src.utils.tensors import collate
from src.datasets.humanact12poses import HumanAct12Poses
from src.utils.get_model_and_data import get_model_and_data
from src.parser.generate import parser

sys.path.append(dtw_folder)
from soft_dtw_cuda import SoftDTW

sys.path.append(deep_edit_folder)
from style_transfer.networks import EncoderStyle, JointGen
from style_transfer.config import Config


class LowerBodyScore(nn.Module):
    def __init__(self):
        super(LowerBodyScore, self).__init__()
        self.ignore_joints = [3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
        self.dtw = SoftDTW(use_cuda=False, dist_func=self.distance)

    def distance(self, x, y):
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def mask(self, data):
        for j in self.ignore_joints:
            data[j, :, :] = 0
        return data

    def forward(self, x, y):
        f1 = torch.squeeze(x["output_xyz"], 0)
        f1 = self.mask(f1)
        f1 = f1.reshape(f1.shape[0] * f1.shape[1], f1.shape[2])
        f1 = f1.permute(1, 0)
        f2 = torch.squeeze(y["output_xyz"], 0)
        f2 = self.mask(f2)
        f2 = f2.reshape(f2.shape[0] * f2.shape[1], f2.shape[2])
        f2 = f2.permute(1, 0)
        d = self.dtw(f1, f2)
        return d


class SkelEmbeddingScore(nn.Module):
    def __init__(self, param, model_path):
        super(SkelEmbeddingScore, self).__init__()
        full_model, _ = get_model_and_data(param)
        state_dict = torch.load(model_path, map_location=param["device"])
        full_model.load_state_dict(state_dict)
        self.model = full_model.encoder.skelEmbedding.to("cuda").eval()
        self.dtw = SoftDTW(use_cuda=False, dist_func=self.distance)

    def distance(self, x, y):
        x = self.model(x)
        y = self.model(y)
        x = torch.squeeze(x, 1)
        y = torch.squeeze(y, 1)
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, x, y):
        bs, njoints, nfeats, nframes = x["output"].shape
        x = x["output"].permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats).to("cuda")
        bs, njoints, nfeats, nframes = y["output"].shape
        y = y["output"].permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats).to("cuda")
        d = self.dtw(x, y)
        return d


class EuclideanScore(nn.Module):
    def __init__(self):
        super(EuclideanScore, self).__init__()
        self.dtw = SoftDTW(use_cuda=False, dist_func=self.distance)

    def distance(self, x, y):
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, x, y):
        f1 = torch.squeeze(x["output_xyz"], 0)
        f1 = f1.reshape(f1.shape[0] * f1.shape[1], f1.shape[2])
        f1 = f1.permute(1, 0)
        f2 = torch.squeeze(y["output_xyz"], 0)
        f2 = f2.reshape(f2.shape[0] * f2.shape[1], f2.shape[2])
        f2 = f2.permute(1, 0)
        d = self.dtw(f1, f2)
        return d


class StyleScore(nn.Module):
    def __init__(self, model_path):
        super(StyleScore, self).__init__()
        cfg = Config()
        state_dict = torch.load(model_path, map_location=cfg.device)
        full_model = JointGen(cfg)
        full_model.load_state_dict(state_dict['gen'])
        self.model = full_model.enc_style3d.to("cuda").eval()

    def calculate_features(self, batch):
        final = self.model(batch)
        return final

    def forward(self, x, y):
        f1 = self.calculate_features(x)
        f2 = self.calculate_features(y)
        d = ((f1 - f2) ** 2).sum()
        return d


class EncoderScore(nn.Module):
    def __init__(self, param, model_path):
        super(EncoderScore, self).__init__()
        full_model, _ = get_model_and_data(param)
        state_dict = torch.load(model_path, map_location=param["device"])
        full_model.load_state_dict(state_dict)
        self.model = full_model.encoder.to("cuda").eval()
        # print(self.model.seqTransEncoder.layers)
        num_layers = len(self.model.seqTransEncoder.layers)
        self.activation = {}
        self.dtw = SoftDTW(use_cuda=True)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def calculate_features(self, batch):
        self.activation = {}
        x, y, mask = batch["x"], batch["y"], batch["mask"].to("cuda")
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats).to("cuda")
        x = self.model.skelEmbedding(x)
        xseq = torch.cat((self.model.muQuery[y][None], self.model.sigmaQuery[y][None], x), axis=0)

        # add positional encoding
        xseq = self.model.sequence_pos_encoder(xseq)

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)
        maskseq = torch.cat((muandsigmaMask, mask), axis=1)

        self.model.seqTransEncoder.layers[1].norm2.register_forward_hook(self.get_activation('layer_1_norm2'))
        final = self.model.seqTransEncoder(xseq, src_key_padding_mask=~maskseq)
        copy = self.activation
        return copy

    def forward(self, x, y):
        # not obvious about whether two objects belong in the same classes
        f1 = self.calculate_features(x)
        f2 = self.calculate_features(y)
        f1d = f1["layer_1_norm2"].permute((1, 0, 2))
        f2d = f2["layer_1_norm2"].permute((1, 0, 2))
        d = self.dtw(f1d, f2d)
        return d


class FIDScore(nn.Module):
    def __init__(self, dataset="humanact12", device="cuda"):
        super(FIDScore, self).__init__()
        dataset_opt = {"ntu13": {"joints_num": 18,
                                 "input_size_raw": 54,
                                 "num_classes": 13},
                       "humanact12": {"input_size_raw": 72,
                                      "joints_num": 24,
                                      "num_classes": 12}}

        param = dataset_opt[dataset]
        self.model = load_classifier_for_fid(dataset, param["input_size_raw"], param["num_classes"], device=device).train()

    def compute_features(self, data):
        with torch.no_grad():
            feat = self.model(data["output_xyz"], data["lengths"])
        return feat

    def normalize(self, v):
        norm = torch.sqrt(torch.sum(v**2))
        return v / (norm + 1e-10)

    def forward(self, x, y):
        with torch.backends.cudnn.flags(enabled=False):
            f1 = self.model(x["output_xyz"], x["lengths"])
            f1 = self.normalize(f1)
            f2 = self.model(y["output_xyz"], y["lengths"])
            f2 = self.normalize(f2)
            d = ((f1 - f2) ** 2).sum()
        return d


class FIDSubsequence(FIDScore):
    def __init__(self, dataset="humanact12", device="cuda"):
        super(FIDSubsequence, self).__init__()
        self.dtw = SoftDTW(use_cuda=False, dist_func=self.distance)

    def distance(self, x, y):
        with torch.backends.cudnn.flags(enabled=False):
            f1 = []
            for s in x:
                fs = self.model(torch.unsqueeze(s, 0), torch.tensor(np.ones(1, dtype=np.int32) * 8, device="cuda"))
                f1.append(fs)
            f2 = []
            for s in y:
                fs = self.model(torch.unsqueeze(s, 0), torch.tensor(np.ones(1, dtype=np.int32) * 8, device="cuda"))
                f2.append(fs)
            f1 = torch.cat(f1)
            f2 = torch.cat(f2)
            f1 = torch.unsqueeze(f1, 0)
            f2 = torch.unsqueeze(f2, 0)
            n = f1.size(1)
            m = f2.size(1)
            d = f1.size(2)
            f1 = f1.unsqueeze(2).expand(-1, n, m, d)
            f2 = f2.unsqueeze(1).expand(-1, n, m, d)
            d = torch.pow(f1 - f2, 2).sum(3)
        return d

    def forward(self, x, y):
        with torch.backends.cudnn.flags(enabled=False):
            bf1 = torch.squeeze(x["output_xyz"], 0).unfold(-1, 8, 4)
            bf1 = bf1.permute(2, 0, 1, 3)
            bf2 = torch.squeeze(y["output_xyz"], 0).unfold(-1, 8, 4)
            bf2 = bf2.permute(2, 0, 1, 3)
            d = self.dtw(bf1, bf2)
        return d


def plot_in_and_out(differences_data, scorer_name="", show=False):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    subplot_titles = ["All differences", "Same class", "Different classes"]
    for ind, data in enumerate(differences_data):
        if len(data) > 0:
            ax[ind].hist(data)
        ax[ind].title.set_text(subplot_titles[ind])
    fig.suptitle(f"Histogram for {scorer_name} distances")
    fig.savefig(f"{scorer_name}.png")

    if show:
        plt.show()


def test_sim_score(scorer, save_data):
    same_class = []
    diff_class = []
    all_diff = []

    start = time()
    for i in tqdm(range(len(save_data)), desc=f"Calculating distance.."):
        # print(save_data[i]["x"].shape)
        for j in range(i + 1, len(save_data)):
            diff = scorer(save_data[i], save_data[j]).item()
            all_diff.append(diff)
            if save_data[i]["y"].item() == save_data[j]["y"].item():
                same_class.append(diff)
            else:
                diff_class.append(diff)
    elapsed = time() - start
    print(f"Took {elapsed}")
    print(f"Average time {elapsed / len(all_diff)}")
    return [all_diff, same_class, diff_class]


def sample_data(loader, min_each=20, num_class=12):
    classes = np.zeros(num_class)
    save_data = []
    for batch in loader:
        if all(classes >= min_each):
            break
        class_name = batch["y"].item()
        if class_name >= num_class:
            continue
        if classes[class_name] >= min_each:
            continue
        classes[class_name] += 1
        save_data.append(batch)
    return save_data


def transform_data(data, device="cuda"):
    from core.actor_wrapper import ActorWrapper
    param, folder, checkpoint, epoch = parser()
    checkpoint_path = os.path.join(folder, checkpoint)
    actor = ActorWrapper(param, checkpoint_path)
    transformed_data = []
    with torch.no_grad():
        for databatch in tqdm(data, desc="Transforming data.."):
            batch = {key: val.to(device) for key, val in databatch.items()}
            batch["x_xyz"] = actor.rot2xyz(batch["x"].to(device),
                                           batch["mask"].to(device))
            batch["output"] = batch["x"]
            batch["output_xyz"] = batch["x_xyz"]
            transformed_data.append(batch)
    return transformed_data


def euc_main(save_data):
    euc = EuclideanScore()
    transformed_data = transform_data(save_data)
    diff_data = test_sim_score(euc, transformed_data)
    plot_in_and_out(diff_data, scorer_name="euc")


def style_main():
    from style_transfer.data_loader import process_single_bvh
    scorer = StyleScore(scorer_default_paths["style"])
    # print(scorer.model.conv_model)
    # return
    cfg = Config()
    test_data = process_single_bvh("/home/linh/projects/deep-motion-editing/style_transfer/data/xia_test/depressed_18_000.bvh", cfg, to_batch=True)
    test_data1 = process_single_bvh("/home/linh/projects/deep-motion-editing/style_transfer/data/xia_test/depressed_13_000.bvh", cfg, to_batch=True)

    print(test_data.keys())
    print(test_data['style3d'].shape)
    print(test_data1['style3d'].shape)
    d = scorer(test_data['style3d'].to("cuda"), test_data1['style3d'].to("cuda"))
    # print(d)


def fid_main(save_data):
    fid = FIDScore()
    transformed_data = transform_data(save_data)
    diff_data = test_sim_score(fid, transformed_data)
    plot_in_and_out(diff_data, scorer_name="fid")


def subsequence_main(save_data):
    scorer = FIDSubsequence()
    transformed_data = transform_data(save_data)
    diff_data = test_sim_score(scorer, transformed_data)
    plot_in_and_out(diff_data, scorer_name="subsequence")


def encoder_main(save_data):
    param, folder, checkpoint, epoch = parser()
    checkpoint_path = os.path.join(folder, checkpoint)
    scorer = EncoderScore(param, checkpoint_path)
    diff_data = test_sim_score(scorer, save_data)
    plot_in_and_out(diff_data, scorer_name="encoder")


def skel_main(save_data):
    param, folder, checkpoint, epoch = parser()
    checkpoint_path = os.path.join(folder, checkpoint)
    scorer = SkelEmbeddingScore(param, checkpoint_path)
    diff_data = test_sim_score(scorer, save_data)
    plot_in_and_out(diff_data, scorer_name="skel")


def make_skel_scorer(checkpoint_path):
    from src.parser.tools import load_args
    folder, checkpoint = os.path.split(checkpoint_path)
    param = load_args(os.path.join(folder, "opt.yaml"))
    param["device"] = "cuda"
    skel = SkelEmbeddingScore(param, checkpoint_path)
    return skel


if __name__ == '__main__':
    # style_main()
    # exit(0)
    min_each = 2
    dataset = HumanAct12Poses(datapath="/home/linh/projects/ACTOR/data/HumanAct12Poses", num_frames=-1)
    iterator = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate)
    save_data = sample_data(iterator, min_each=min_each)
    skel_main(save_data)
    # euc_main(save_data)
    # exit(0)
    # subsequence_main(save_data)
