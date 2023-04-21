#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
import clip

from .paths import *
import sys
sys.path.append(clip_folder)
from clip_src.models import get_model as clip_get_model
from clip_src.models.rotation2xyz import Rotation2xyz
from clip_src.parser.tools import load_args


class ClipEncoder(nn.Module):
    def __init__(self, param, checkpoint_path):
        super(ClipEncoder, self).__init__()
        param["njoints"] = 25
        param["nfeats"] = 6
        param["num_classes"] = 1
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=param['device'], jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()

        for p in clip_model.parameters():
            p.requires_grad = False

        model = clip_get_model.get_model(param, clip_model)
        state_dict = torch.load(checkpoint_path, map_location=param["device"])
        model.load_state_dict(state_dict, strict=False)
        self.model = model.encoder.to(param["device"]).eval()
        self.activation = {}

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.clone()
        return hook

    def calculate_features(self, batch):
        self.activation = {}
        x = batch["output"]
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        x = self.model.skelEmbedding(x)
        xseq = torch.cat((self.model.muQuery[[0]][None], self.model.sigmaQuery[[0]][None], x), axis=0)
        xseq = self.model.sequence_pos_encoder(xseq)
        self.model.seqTransEncoder.layers[1].norm2.register_forward_hook(self.get_activation('layer_1_norm_2'))
        self.model.seqTransEncoder.layers[2].norm2.register_forward_hook(self.get_activation('layer_2_norm_2'))
        self.model.seqTransEncoder.layers[3].norm2.register_forward_hook(self.get_activation('layer_3_norm_2'))
        final = self.model.seqTransEncoder(xseq)
        copy = self.activation
        return copy

    def forward(self, x, y):
        f1 = self.calculate_features(x)
        f2 = self.calculate_features(y)
        d = 0
        for k in f1.keys():
            d += ((f1[k] - f2[k]) ** 2).sum()
        return d


class ClipWrapper():
    def __init__(self, param, checkpoint_path):
        param["njoints"] = 25
        param["nfeats"] = 6
        param["num_classes"] = 1
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=param['device'], jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        model = clip_get_model.get_model(param, clip_model)
        state_dict = torch.load(checkpoint_path, map_location=param["device"])
        model.load_state_dict(state_dict, strict=False)
        self.model = model.decoder.eval()
        self.model.seqTransDecoder.cuda().eval()
        self.model.seqTransDecoder.requires_grad_(False)
        self.model.finallayer.requires_grad_(False)
        self.param = param
        self.latent_dim = param["latent_dim"]
        self.device = param["device"]
        self.clip_model = model.clip_model
        self.clip_model.training = False
        self.rotator = Rotation2xyz(device=param["device"])

        # from humanact12
        self.class_texts = {
            0: "warm_up",
            1: "walk",
            2: "run",
            3: "jump",
            4: "drink",
            5: "lift_dumbbell",
            6: "sit",
            7: "eat",
            8: "turn steering wheel",
            9: "phone",
            10: "boxing",
            11: "throw",
        }
        self.class_embeddings = {}
        for k in self.class_texts.keys():
            token = clip.tokenize(self.class_texts[k]).to(param["device"])
            self.class_embeddings[k] = self.clip_model.encode_text(token).float()

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def sample_vector(self, sampn=1, sample_class=0, random_alpha=0.001):
        import numpy as np
        # how to pass the clip features later?
        if sample_class is None:
            sample_class = np.random.randint(12)
        refvec = torch.randn((sampn, self.latent_dim)).to(self.device) * random_alpha + self.class_embeddings[sample_class]
        return refvec

    def generate(self, z, durations=torch.ones((1, 1), dtype=torch.int) * 60):
        if len(durations.shape) == 1:
            lengths = durations.to(self.device)
        else:
            lengths = durations.to(self.device).reshape(z.shape[0])
        mask = self.lengths_to_mask(lengths)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.param["njoints"], self.param["nfeats"]
        z = z[None]
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.model.sequence_pos_encoder(timequeries)
        output = self.model.seqTransDecoder(tgt=timequeries, memory=z,
                                            tgt_key_padding_mask=~mask)
        output = self.model.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)
        batch = {}
        batch["output"] = output
        batch["output"][:, 0] = torch.tensor([1, 0, 0, 0, -1, 0]).unsqueeze(0).unsqueeze(2)
        batch["output_xyz"] = self.rotator(output, None,
                                           self.param["pose_rep"],
                                           self.param["translation"],
                                           self.param["glob"],
                                           "smpl",
                                           vertstrans=False, beta=0,
                                           glob_rot=self.param["glob_rot"])
        # for motion clip visualization
        batch["lengths"] = lengths
        batch["y"] = torch.ones((1, 1))
        return batch


def make_clip_wrapper(checkpoint_path):
    folder, checkpoint = os.path.split(checkpoint_path)
    param = load_args(os.path.join(folder, "opt.yaml"))
    param["device"] = "cuda"
    clip = ClipWrapper(param, checkpoint_path)
    return clip


def make_clip_scorer(checkpoint_path):
    folder, checkpoint = os.path.split(checkpoint_path)
    param = load_args(os.path.join(folder, "opt.yaml"))
    param["device"] = "cuda"
    clip = ClipEncoder(param, checkpoint_path)
    return clip


if __name__ == '__main__':
    from .visualization import render, plot_3d
    from time import time
    start = time()
    clipper = make_clip_wrapper(checkpoint_default_paths["clip"])
    loaded = time()
    print(f"Loading took {loaded - start}")
    sample_class = 2
    z = clipper.sample_vector(sample_class=sample_class)
    gen = clipper.generate(z, durations=torch.ones((1, 1), dtype=int) * 60)
    generated = time()
    print(f"Generation took {generated - loaded}")
    render(gen, clipper.param, f"{DEFAULT_SAVE_FOLDER}/clip_gen_class_{sample_class}.gif")
    print(f"Rendering took {time() - generated}")
