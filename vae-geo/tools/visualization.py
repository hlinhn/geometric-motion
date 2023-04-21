#!/usr/bin/env python3

import os
import torch
import sys
import numpy as np
from PIL import Image

from .paths import *
sys.path.append(actor_folder)
from src.render.renderer import get_renderer
from src.render.rendermotion import render_video
from src.models.rotation2xyz import Rotation2xyz

sys.path.append(clip_folder)
from clip_src.visualize.anim import plot_3d_motion


def render(data, param, name="test_vid.gif", save_folder=DEFAULT_SAVE_FOLDER, class_name="unknown"):
    rotator = Rotation2xyz(device="cuda")
    xyz_gen = rotator(data["output"], None,
                      param["pose_rep"],
                      param["translation"],
                      param["glob"],
                      "vertices",
                      vertstrans=False, beta=0,
                      glob_rot=param["glob_rot"])

    width = 256
    height = 256
    background = np.zeros((height, width, 3))
    renderer = get_renderer(width, height)
    path = os.path.join(save_folder, name)
    mesh = xyz_gen.cpu().detach().numpy()[0].transpose(2, 0, 1)
    render_video(mesh, "random", class_name, renderer, path, background)


def unroll_gif(file_name, savename):
    selected = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 59]
    images = []
    with Image.open(file_name) as im:
        for i in selected:
            im.seek(i)
            image = im.convert("RGBA")
            datas = image.getdata()
            newData = []
            for item in datas:
                if item[3] == 0:  # if transparent
                    newData.append(trans_color)  # set transparent color in jpg
                else:
                    newData.append(tuple(item[:3]))
            image = Image.new("RGB", im.size)
            image.getdata()
            image.putdata(newData)
            images.append(image)
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(savename)


def plot_3d(data, name="test_vid.gif", save_folder=DEFAULT_SAVE_FOLDER):
    save_path = os.path.join(save_folder, name)
    param = {}
    param["fps"] = 20
    param["pose_rep"] = "xyz"
    param["appearance_mode"] = "motionclip"
    plot_3d_motion(data["output_xyz"].cpu().numpy()[0], data["lengths"].cpu().numpy()[0], save_path, param)


if __name__ == '__main__':
    unroll_gif(sys.argv[1], sys.argv[2])
