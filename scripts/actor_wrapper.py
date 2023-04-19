#!/usr/bin/env python3


import torch
import os
import numpy as np

import sys
sys.path.append('/home/linh/projects/ACTOR')
from src.utils import get_model_and_data as actor_get_model_and_data
from src.models.rotation2xyz import Rotation2xyz
from src.parser.tools import load_args

class ActorWrapper():
    def __init__(self, param, checkpoint_path):
        self.param = param
        full_model, _ = actor_get_model_and_data.get_model_and_data(param)
        state_dict = torch.load(checkpoint_path, map_location=param["device"])
        full_model.load_state_dict(state_dict)
        self.model = full_model.decoder.eval()
        self.model.seqTransDecoder.cuda().eval()
        self.model.seqTransDecoder.requires_grad_(False)
        self.model.finallayer.requires_grad_(False)
        self.latent_dim = param["latent_dim"]
        self.rotator = Rotation2xyz(device=param["device"])

    def sample_vector(self, sampn=1, sample_class=None, device="cuda"):
        refvec = torch.randn((sampn, self.latent_dim)).to(device)
        if sample_class is not None:
            bias = self.model.actionBiases[sample_class]
            refvec = torch.add(refvec, bias)
        else:
            combi = torch.rand(12).to(device)
            combi = combi / torch.sum(combi)
            combi = combi[:, None]
            bias = torch.sum(combi * self.model.actionBiases, axis=0)
            refvec = torch.add(refvec, bias)
        return refvec

    def rot2xyz(self, data, mask, jointstype="smpl"):
        return self.rotator(data, None,
                            self.param["pose_rep"],
                            self.param["translation"],
                            self.param["glob"],
                            jointstype,
                            vertstrans=True, beta=0,
                            glob_rot=self.param["glob_rot"])

    def generate(self, z, jointstype="smpl", nframes=60):
        bs = 1
        timequeries = torch.zeros(nframes, bs, z.shape[1], device=z.device)
        time_series = self.model.sequence_pos_encoder(timequeries)
        output = self.model.seqTransDecoder(tgt=time_series, memory=z)
        output = self.model.finallayer(output).reshape(nframes, bs, self.model.njoints, self.model.nfeats)
        output = output.permute(1, 2, 3, 0)
        result = {}
        result["output"] = output
        result["output_xyz"] = self.rotator(output, None,
                                            self.param["pose_rep"],
                                            self.param["translation"],
                                            self.param["glob"],
                                            jointstype,
                                            vertstrans=False, beta=0,
                                            glob_rot=self.param["glob_rot"])
        length_data = np.ones(1, dtype=np.int32) * nframes
        result["lengths"] = torch.tensor(length_data).to(z.device)

        return result


def make_actor_wrapper(checkpoint_path):
    folder, checkpoint = os.path.split(checkpoint_path)
    param = load_args(os.path.join(folder, "opt.yaml"))
    param["device"] = "cuda"
    actor = ActorWrapper(param, checkpoint_path)
    return actor


def test_wrapper():
    actor = make_actor_wrapper(sys.argv[1])
    print(actor.param)
    ref = actor.sample_vector()
    res = actor.generate(ref)
    print(res["output"].shape)
    # render(res, actor)


if __name__ == '__main__':
    test_wrapper()
    # make_actor_wrapper()
