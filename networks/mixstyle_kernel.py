import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.svm import SVC
from torch.distributions import kl_divergence

class SaveMuVar():
    mu, var = None, None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.mu = output.detach().cpu().mean(dim=[2, 3], keepdim=True).squeeze(-1).squeeze(-1).numpy()
        self.var = output.detach().cpu().var(dim=[2, 3], keepdim=True).squeeze(-1).squeeze(-1).numpy()

    def remove(self):
        self.hook.remove()


class EFDMix(nn.Module):
    """EFDMix.

    Reference:
      Zhang et al. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. CVPR 2022.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        x_view = x.view(B, C, -1)
        value_x, index_x = torch.sort(x_view)  # sort inputs
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index)
        new_x = x_view + (x_view_copy - x_view.detach()) * (1-lmda)
        return new_x.view(B, C, W, H)


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True  # Train: True, Test: False

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)
        return x_normed*sig_mix + mu_mix


class TriD(nn.Module):
    """TriD.
    Reference:
      Chen et al. Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation. MICCAI 2023.
    """
    def __init__(self, p=0.5, eps=1e-6, alpha=0.1):
        """
        Args:
          p (float): probability of using TriD.
          eps (float): scaling parameter to avoid numerical issues.
          alpha (float): parameter of the Beta distribution.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self._activated = True  # Train: True, Test: False
        self.beta = torch.distributions.Beta(alpha, alpha)

    def set_activation_status(self, status=True):
        self._activated = status




    def forward(self, x):
        if not self._activated:
            return x

        if random.random() > self.p:
            return x

        N, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True)

        var = x.var(dim=[2, 3], keepdim=True)


        if random.random() > 0.5:# 参数0.8代表随机选择的概率，可以调整，经过试验下来，发现为0.5效果较好

        # #     # 假设 input 是一个张量对象
            swap_index_cpu = torch.randperm(x.size(0))

            # 将生成的随机排列索引转移到 CUDA 设备上
            swap_index = swap_index_cpu.to(x.device)


            swap_mean = mu[swap_index]
            swap_std = var[swap_index]

            scale = swap_std / var
            shift = swap_mean - mu * scale
            output = x * scale + shift
        else:
            # print('aug randomly choice 选择TRID')
            # print('trid randomly choice:',random.random())
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            lmda = self.beta.sample((N, C, 1, 1))
            bernoulli = torch.bernoulli(lmda).to(x.device)
            # 获取当前数据分布，并根据目标数据分布更新混合比例
            # mixing_ratio = self.update_mixing_ratio()

            # 使用混合比例进行操作 高斯分布
            # 在 CPU 上生成随机数
            mu_random_cpu = torch.randn((N, C, 1, 1), dtype=torch.float32, device='cpu')

            # 将生成的随机数转移到 CUDA 设备上
            mu_random= mu_random_cpu.to(x.device)
            # mu_random = torch.randn((N, C, 1, 1), dtype=torch.float32).to(x.device)
            var_random_cpu = torch.randn((N, C, 1, 1), dtype=torch.float32,device='cpu')
            var_random = var_random_cpu.to(x.device)
            mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
            sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
            output=x_normed * sig_mix + mu_mix

        return output


