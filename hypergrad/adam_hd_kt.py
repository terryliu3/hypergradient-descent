import math
import torch
from torch.optim.optimizer import Optimizer


class AdamHDKT(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        wealth (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_

    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Online Learning Rate Adaptation with Hypergradient Descent:
        https://openreview.net/forum?id=BkrsAzWAb
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, wealth=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, wealth=wealth)
        super(AdamHDKT, self).__init__(params, defaults)
        
        self._step = 0
        # Keep a copy of the very first learning rate
        self._lr0 = lr
        self._sum_of_normalized_hypergrads = 0.0
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Global initialization
            group['hypergrad'] = 0.0
            group['squared_norm_u'] = 0.0
            group['squared_norm_v'] = 0.0
            
            self._step += 1
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                if self._step > 1:
                    prev_bias_correction1 = 1 - beta1 ** (self._step - 1)
                    prev_bias_correction2 = 1 - beta2 ** (self._step - 1)
                    # Hypergradient for Adam:
                    u = grad.view(-1)
                    v = torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).view(-1) * math.sqrt(prev_bias_correction2) / prev_bias_correction1
                    h = -torch.dot(u, v) 
                    group['hypergrad'] += h.item()
                    group['squared_norm_u'] += (u.norm()**2).item()
                    group['squared_norm_v'] += (v.norm()**2).item()
                    # # Update dual vector
                    # state['sum_of_hypergrads'] += h.item()
                    # # Update wealth
                    # state['wealth'] += -h.item() * (group['lr'] - state['lr0'])
                    # # Update learning rate
                    # group['lr'] = state['wealth'] * (-state['sum_of_hypergrads']) / state['step'] + state['lr0']
                    
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** self._step
                bias_correction2 = 1 - beta2 ** self._step
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
            
            if self._step > 1:
                group['normalized_hypergrad'] = group['hypergrad'] / math.sqrt(group['squared_norm_u'] * group['squared_norm_v'] + 1e-12)
                # Update dual vector

                self._sum_of_normalized_hypergrads += group['normalized_hypergrad']
                # Update wealth
                group['wealth'] += -group['normalized_hypergrad'] * (group['lr'] - self._lr0)
                # Update learning rate
                group['lr'] = group['wealth'] * (-self._sum_of_normalized_hypergrads) / self._step + self._lr0
                
        return loss
