#!/usr/bin/env python3

import os


PROJECT_FOLDER = '/home/linh/projects'
DEFAULT_SAVE_FOLDER = '/home/linh/vae-geo'
geometry_folder = os.path.join(PROJECT_FOLDER, 'GAN_Geometry')
actor_folder = os.path.join(PROJECT_FOLDER, 'ACTOR')
clip_folder = os.path.join(PROJECT_FOLDER, 'MotionCLIP')
dtw_folder = os.path.join(PROJECT_FOLDER, 'pytorch-softdtw-cuda')
deep_edit_folder = os.path.join(PROJECT_FOLDER, 'deep-motion-editing')

checkpoint_default_paths = {
    "actor": f"{actor_folder}/pretrained_models/humanact12/checkpoint_5000.pth.tar",
    "clip": "/home/linh/Downloads/motionclip/paper-model/checkpoint_0100.pth.tar"
}
scorer_default_paths = {
    "encoder": "/home/linh/Downloads/motionclip/paper-model/checkpoint_0100.pth.tar",
    "skel": f"{actor_folder}/pretrained_models/humanact12/checkpoint_5000.pth.tar",
    "style": f"{deep_edit_folder}/style_transfer/pretrained/pth/gen_00100000.pt"
}

HUMANACT12_DATAPATH = f"{actor_folder}/data/HumanAct12Poses"

available_dist_functions = ["fid", "subseq", "encoder", "euc", "skel", "low"]
available_wrappers = ["actor", "clip"]
