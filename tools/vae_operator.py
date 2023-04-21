#!/usr/bin/env python3

from time import time
import numpy as np
import sys
import torch

from .paths import *
sys.path.append(geometry_folder)

from core.GAN_hvp_operator import Operator
from core.lanczos_generalized import lanczos


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


def vae_hessian(G, feat, dist, device="cuda", cut_off=None, print_time=False):
    start = time()
    metrichvp = VAEHVPOperator(G, feat, dist)
    if not cut_off:
        cut_off = feat.numel() // 2 - 1
    eigvals, eigvects = lanczos(metrichvp, num_eigenthings=cut_off, use_gpu=True)
    eigvects = eigvects.T
    H = eigvects @ np.diag(eigvals) @ eigvects.T
    if print_time:
        print(f"Took {time() - start}")
    return eigvals, eigvects, H


def get_full_hessian(l, param):
    hs = param.numel()
    hessian = torch.zeros(hs, hs)
    loss_grad = torch.autograd.grad(l, param, create_graph=True, retain_graph=True, only_inputs=True)[0].view(-1)
    for idx in range(hs):
        grad2rd = torch.autograd.grad(loss_grad[idx], param, create_graph=False, retain_graph=True, only_inputs=True)
        hessian[idx] = grad2rd[0].view(-1)
    return hessian.cpu().data.numpy()


def calculate_full_hessian(G, feat, dist, device="cuda", print_time=True):
    start = time()
    ref_vec = feat.detach().clone().float().to(device)
    mov = ref_vec.float().detach().clone().requires_grad_(True)
    mov = mov
    v1 = G.generate(ref_vec)
    v2 = G.generate(mov)
    d = dist(v1, v2)
    H = get_full_hessian(d, mov)
    eigval, eigvec = np.linalg.eigh(H)
    if print_time:
        print(f"Took {time() - start}")
    return eigval, eigvec, H
