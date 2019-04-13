import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Optimizer
from torch import nn
import torch.nn.functional as F
from kfac.approx_fisher import *

class KFAC(Optimizer):
    def __init__(self, model, loss_func, approx_type='diagonal', lam=0.2, eta=1e-5, T_2=20, update_gamma_every=20,
                 update_lam_every=5, update_approx_every=20, sample_ratio=1/8):
        self.kfac_params = {}
        self.kfac_params['lam'] = lam
        self.kfac_params['eta'] = eta
        self.kfac_params['eps'] = 0
        self.gamma = np.sqrt(lam + eta)
        omega = np.sqrt(19 / 20) ** T_2
        self.gamma_coef = [1, omega, 1 / omega]
        self.n_iter = 1
        self.sample_ratio = sample_ratio
        self.update_gamma_every = update_gamma_every
        self.update_lam_every = update_lam_every
        self.update_approx_every = update_approx_every

        self.model = model
        self.layers = [layer for layer in model.children()]
        self.net_input = None
        self.final_preacts = None
        self.log_output_dist = None
        self.prev_loss = None
        self.loss_func = loss_func
        self.update = None

        self.layers[0].register_forward_hook(self.get_net_input)
        self.layers[-1].register_forward_hook(self.get_net_output)

        if approx_type == 'diagonal':
            self.approx_F = BlockDiagonalFisher(model, self.kfac_params)
        # Add support for tridiagonal Fisher later
        else:
            pass

    def step(self, batch_loss):
        grads = self.get_grads()
        # Backprop with targets sampled from the output distribution to calculate
        # the necessary statistics for the Fisher Blocks
        self.sample_backprop()

        gammas = self.get_canditate_gammas()
        self.kfac_params['eps'] = min(1 - 1 / self.n_iter, 0.95)

        proposals = []
        for gamma in gammas:
            if not self.n_iter % self.update_approx_every or self.n_iter < 4:
                self.approx_F.damp(gamma)
                self.approx_F.invert()

            proposals.append(torch.unsqueeze(-self.approx_F.inv_v_prod(grads), 1))

        proposals = torch.cat(proposals, dim=1)
        alpha, M = self.calc_quad_approx_params(proposals, grads, batch_loss)

        opt_index = torch.argmin(M)
        opt_proposal = proposals[:, opt_index]
        self.M_min = M[opt_index]
        self.update = opt_proposal * alpha[opt_index]
        self.update_params(self.update)
        self.gamma = gammas[opt_index]

        if self.n_iter > 1 and not (self.n_iter - 1) % self.update_lam_every:
            self.update_lambda(batch_loss)

        self.prev_loss = batch_loss
        self.n_iter += 1

    def update_lambda(self, batch_loss):
        loss_delta = self.prev_loss - batch_loss
        rho = loss_delta / -self.M_min

        if rho > 0.75:
            self.kfac_params['lam'] = self.gamma_coef[1] * self.kfac_params['lam']
        elif rho < 0.25:
            self.kfac_params['lam'] = 1 / self.gamma_coef[2] * self.kfac_params['lam']

    def get_canditate_gammas(self):
        if not self.n_iter % self.update_approx_every or self.n_iter == 1:
            return [self.gamma * coef for coef in self.gamma_coef]

        return [self.gamma]

    def calc_quad_approx_params(self, proposal, grads, batch_loss):
        flat_grads = torch.cat([grad.flatten() for grad in grads])
        grads_dot_proposal = torch.matmul(flat_grads, proposal)
        exact_fisher_prod = self.exact_fisher_prod(proposal) + \
            (self.kfac_params['eta'] + self.kfac_params['lam']) * torch.norm(proposal, p=2, dim=0) ** 2
        alpha = -grads_dot_proposal / exact_fisher_prod
        M = alpha ** 2 / 2 + alpha * grads_dot_proposal + batch_loss

        return alpha, M

    def exact_fisher_prod(self, v):
        jvp = self.jvp(v)

        batch_size = jvp.size()[0]
        n_outputs = jvp.size()[1]

        p = torch.exp(self.log_output_dist)
        q = torch.sqrt(p)
        q_diag = torch.diag_embed(q)
        B = q_diag - torch.einsum('ij,ik->ijk', p, q)
        F_R = torch.einsum('ijk,ijl->ikl', B.permute(0, 2, 1), B)
        right_prod = torch.einsum('ijk,ikl->ijl', F_R, jvp)
        F_prod = torch.einsum('ijk,ijk->ik', jvp, right_prod)

        return torch.mean(F_prod, dim=0)

    # Right multiply the Jacobian of the network outputs w.r.t. the weights
    # by a vector
    def jvp(self, v):
        if len(v.size()) > 1:
            if self.log_output_dist.is_cuda:
                jvp = torch.zeros(*self.log_output_dist.size(), v.size()[1],
                    device=self.log_output_dist.get_device())
            else:
                jvp = jvp = torch.zeros(*self.log_output_dist.size(), v.size()[1])
        else:
            if self.log_output_dist.is_cuda:
                jvp = torch.zeros(*self.log_output_dist.size(),
                    device=self.log_output_dist.get_device())
            else:
                jvp = jvp = torch.zeros(*self.log_output_dist.size())

        for i in range(jvp.size()[0]):
            for j in range(jvp.size()[1]):
                grads_unrolled = []
                self.model.zero_grad()
                # Get the row of the Jacobian for the ith batch corresponding to the jth output
                grads = torch.autograd.grad(self.final_preacts[i][j], self.model.parameters(), retain_graph=True)
                for k in range(len(grads) // 2):
                    dW = grads[k * 2]
                    if len(dW.size()) > 2:
                        dW = dW.reshape(dW.size()[0], -1)
                    db = torch.unsqueeze(grads[k * 2 + 1], 1)
                    grads_unrolled.append(torch.cat([dW, db], dim=1).flatten())

                grads_unrolled = torch.cat(grads_unrolled)
                jvp[i][j] = torch.matmul(grads_unrolled, v)

        self.model.zero_grad()

        return jvp

    def sample_backprop(self):
        self.model.zero_grad()
        batch_size = self.log_output_dist.size()[0]
        sample_size = int(self.sample_ratio * batch_size)
        sample_start = np.random.randint(batch_size - sample_size)
        sample_slice = slice(sample_start, sample_start + sample_size)
        # Save the sample slice in the shared parameter dictionary so we don't save tensors of all
        # 0's in the G update hook
        self.kfac_params['sample slice'] = sample_slice
        sample_outputs = self.sample_output_dist().flatten()
        self.approx_F.set_sampling_flag()
        sample_loss = self.loss_func(self.log_output_dist[sample_slice], sample_outputs)
        sample_loss.backward(retain_graph=True)
        self.approx_F.unset_sampling_flag()

    # Sample the output distribution of the network
    def sample_output_dist(self):
        sample_slice = self.kfac_params['sample slice']
        return torch.multinomial(torch.exp(self.log_output_dist[sample_slice]), 1)

    def get_net_input(self, module, inputs, outputs):
        self.net_input = inputs[0]

    def get_net_output(self, module, inputs, outputs):
        self.final_preacts = outputs
        self.log_output_dist = F.log_softmax(outputs, dim=1)

    def get_grads(self):
        grads = []

        for layer in self.layers:
            params = tuple(layer.parameters())
            dW = params[0].grad
            if type(layer) is nn.Conv2d:
                dW = dW.reshape(dW.size()[0], -1)
            db = torch.unsqueeze(params[1].grad, 1)
            #grads.append(torch.cat([dW, db], dim=1) / self.kfac_params['batch size'])
            grads.append(torch.cat([dW, db], dim=1))

        return grads

    def update_params(self, update):
        i = 0
        for layer in self.layers:
            W, b = tuple(layer.parameters())
            n_weights = W.numel() + b.numel()
            layer_update = update[i:i + n_weights].reshape(W.size()[0], -1)
            W_update = layer_update[:, :-1]
            b_update = layer_update[:, -1]
            if type(layer) is nn.Conv2d:
                W_update = W_update.reshape(*W.size())

            W.data.add_(W_update)
            b.data.add_(b_update)
            i += n_weights

    def zero_grad(self):
        self.model.zero_grad()
