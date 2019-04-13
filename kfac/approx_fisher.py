import numpy as np
import torch
from torch import nn

class BlockDiagonalFisher:
    def __init__(self, model, kfac_params):
        self.blocks = [DiagonalBlock(layer, kfac_params) for layer in model.children()]

    # Right multiply the approximate inverse Fisher by the gradients of the loss function
    def inv_v_prod(self, grads):
        u = []

        for block, grad in zip(self.blocks, grads):
            u.append(block.inv_v_prod(grad).flatten())

        return torch.cat(u)

    # Dampen te blocks
    def damp(self, gamma):
        for block in self.blocks:
            block.damp(gamma)

    # Invert all of the blocks in the approximate Fisher
    def invert(self):
        for block in self.blocks:
            block.invert()

    # Set the sampling flag in each block. The sampling flat is used to determine whether we should
    # save statistics when using the forward and backward methods of the model
    def set_sampling_flag(self):
        for block in self.blocks:
            block.set_sampling_flag()

    def unset_sampling_flag(self):
        for block in self.blocks:
            block.unset_sampling_flag()

    def set_ignore_input_flag(self):
        for block in self.blocks:
            block.set_ignore_input_flag()

    def unset_ignore_input_flag(self):
        for block in self.blocks:
            block.unset_ignore_input_flag()

class DiagonalBlock:
    def __init__(self, layer, kfac_params):
        self.kfac_params = kfac_params
        self.A = 0
        self.G = 0
        self.A_damp_inv = None
        self.G_dam_inv = None
        self.A_eye = None
        self.G_eye = None
        # Register PyTorch hooks so we can save the intermediate layer inputs and gradients
        layer.register_forward_hook(self.forward_hook)
        layer.register_backward_hook(self.backward_hook)
        self.sampling_mode = False
        self.ignore_input = False
        self.jvp_mode = False

    def forward_hook(self, module, inputs, outputs):
        if not self.ignore_input:
            is_cuda = inputs[0].is_cuda

            batch_size = inputs[0].size()[0]
            self.kfac_params['batch size'] = batch_size
            if type(module) is nn.Linear:
                ones = torch.ones(batch_size, 1)
                if is_cuda:
                    ones = ones.to(inputs[0].get_device())
                a = torch.cat([inputs[0].data, ones], dim=1)
            elif type(module) is nn.Conv2d:
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size
                stride = module.stride

                # Extract the image patches from the input
                a = inputs[0].unfold(3, kernel_size[0], stride[0]).data
                a = a.unfold(2, kernel_size[1], stride[1])
                a = a.permute(0, 1, 2, 3, 5, 4)
                a = a.reshape(-1, kernel_size[0] * kernel_size[1])

                ones = torch.ones(a.size()[0], 1)
                if is_cuda:
                    ones = ones.to(inputs[0].get_device())

                a = torch.cat([*[a for c in range(in_channels)], ones], dim=1)

            eps = self.kfac_params['eps']
            A_temp = torch.matmul(torch.t(a), a) / a.size()[0]
            self.A = eps * self.A + (1 - eps) * A_temp

            if self.A_eye is None:
                self.A_eye = torch.eye(self.A.size()[0])
                if self.A.is_cuda:
                    self.A_eye = self.A_eye.to(self.A.get_device())

    def backward_hook(self, module, grad_input, grad_output):
        if self.sampling_mode:
            sample_slice = self.kfac_params['sample slice']
            if type(module) is nn.Linear:
                g = grad_output[0][sample_slice].data
            elif type(module) is nn.Conv2d:
                g = grad_output[0][sample_slice].data
                g = g.permute(0, 2, 3, 1)
                g = g.reshape(-1, g.size()[-1])

            eps = self.kfac_params['eps']
            G_temp = torch.matmul(torch.t(g), g) #/ g.size()[0]
            self.G = eps * self.G + (1 - eps) * G_temp

            if self.G_eye is None:
                self.G_eye = torch.eye(self.G.size()[0])
                if self.G.is_cuda:
                    self.G_eye = self.G_eye.to(self.G.get_device())

    # Right multiply inverse block by a vector and return "rolled up" representation
    def inv_v_prod(self, V):
        U = torch.matmul(torch.matmul(self.G_damp_inv, V), self.A_damp_inv)
        return U

    # Calculate and save the new damping parameters
    def damp(self, gamma):
        A_size = self.A.size()[0]
        G_size = self.G.size()[0]
        self.kfac_params['gamma'] = gamma
        self.pi = torch.sqrt((torch.trace(self.A) / A_size) / (torch.trace(self.G) / G_size))

    def invert(self):
        gamma = self.kfac_params['gamma']
        A_damp = torch.add(self.A, self.pi * gamma * self.A_eye)
        G_damp = torch.add(self.G, 1 / self.pi * gamma * self.G_eye)
        self.A_damp_inv = torch.inverse(A_damp)
        self.G_damp_inv = torch.inverse(G_damp)

    def set_sampling_flag(self):
        self.sampling_mode = True

    def unset_sampling_flag(self):
        self.sampling_mode = False

    def set_ignore_input_flag(self):
        self.ignore_input = True

    def unset_ignore_input_flag(self):
        self.ignore_input = False
