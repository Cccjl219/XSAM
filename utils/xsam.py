import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

class XSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, steps, rho, rho_max, alpha, alpha_max, alpha_delta, eps, **kwargs):

        super(XSAM, self).__init__(params, dict(**kwargs))
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.steps = steps
        self.rho = rho
        self.rho_max = rho_max
        self.alpha = alpha
        self.alpha_max = alpha_max
        self.alpha_delta = alpha_delta
        self.eps = eps
        self.v0_norm = None
        self.latest_grad_norm = None
        self.theta = None
        self.max_loss = None
        self.scale = None

    @torch.no_grad()
    def calculate_latest_grad_norm(self):
        self.latest_grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)

    @torch.no_grad()
    def first_step(self, step):
        self.calculate_latest_grad_norm()
        step_size = self.rho / (self.latest_grad_norm + self.eps)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.data += p.grad * step_size  # ascend to the local maximum "w + e(w)"

    @torch.no_grad()
    def calculate_v0(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self.state[p]["v0"] = p.data - self.state[p]["data_0"]
        self.v0_norm = torch.norm(torch.stack([self.state[p]["v0"].norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self.state[p]["v0"] /= (self.v0_norm + self.eps)

    @torch.no_grad()
    def calculate_v1(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self.state[p]["v1"] = p.grad / (self.latest_grad_norm + self.eps)

    @torch.no_grad()
    def get_v0_v1_theta(self):
        dot = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    dot += (self.state[p]["v0"] * self.state[p]["v1"]).sum()
        return torch.arccos(dot)

    @torch.no_grad()
    def set_grad_alpha_interpolation_of_v0_v1_scaled(self, alpha, scale):
        theta = self.get_v0_v1_theta()
        self.theta = theta
        sin_theta = torch.sin(theta)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = torch.where(sin_theta < 1e-8, (scale * 0.5) * (self.state[p]["v0"] + self.state[p]["v1"]), (scale * torch.sin((1 - alpha) * theta) / sin_theta) * self.state[p]["v0"] + (scale * torch.sin(alpha * theta) / sin_theta) * self.state[p]["v1"])

    @torch.no_grad()
    def set_grad_normalize_scaled(self, scale):
        norm = torch.norm(torch.stack([p.grad.norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad *= scale / (norm + self.eps)

    @torch.no_grad()
    def second_step(self, update_alpha, model, inputs, targets, criterion):
        self.calculate_latest_grad_norm()
        self.scale = self.latest_grad_norm
        self.calculate_v0()
        self.calculate_v1()
        if update_alpha:
            self.update_alpha(model, inputs, targets, criterion)
        self.set_grad_alpha_interpolation_of_v0_v1_scaled(self.alpha, self.scale)
        self.restore_param()
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        return self.scale

    @torch.no_grad()
    def set_param_x_y(self, x, y, theta):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.data = self.state[p]["data_0"] + x / torch.sin(theta) * self.state[p]["v1"] + (y - x * torch.cos(theta) / torch.sin(theta)) * self.state[p]["v0"]

    @torch.no_grad()
    def set_param_rho_max(self, alpha, rho_max):
        self.set_grad_alpha_interpolation_of_v0_v1_scaled(alpha, rho_max)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.data = self.state[p]["data_0"] + p.grad

    @torch.no_grad()
    def update_alpha(self, model, inputs, targets, criterion):
        best_alpha = 0
        self.max_loss = 0
        batch_loss_alpha_list = []
        for alpha in np.arange(0.0, self.alpha_max + 1e-5, self.alpha_delta):
            self.set_param_rho_max(alpha, self.rho_max)
            outputs = model(inputs)
            batch_loss_alpha = criterion(outputs, targets)
            if batch_loss_alpha.item() > self.max_loss:
                self.max_loss = batch_loss_alpha.item()
                best_alpha = alpha
            batch_loss_alpha_list.append(batch_loss_alpha.item())
        self.alpha = best_alpha

    @torch.no_grad()
    def backup_param(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["data_0"] = p.data.clone()

    @torch.no_grad()
    def restore_param(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["data_0"].clone()

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

