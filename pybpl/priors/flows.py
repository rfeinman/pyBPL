import types
import copy
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data



class FlowDensityEstimator(object):


    def __init__(self,X,num_blocks=5,num_hidden=64,lr=.0001,
                batch_size=100,test_batch_size=1000,log_interval=1000,seed=1):


        self.num_blocks=num_blocks
        self.num_hidden=num_hidden
        self.lr=lr
        self.batch_size=batch_size
        self.test_batch_size=test_batch_size
        self.log_interval=log_interval
        self.seed=seed
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)

        self.kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}

        train_split = int(X.shape[0]*(4.0/5.0))
        valid_size = int(X.shape[0]*(1.0/10.0))
        valid_split = train_split+valid_size

        train_numpy = X[:train_split]
        valid_numpy = X[train_split:valid_split]
        test_numpy = X[valid_split:]

        self.train_tensor = torch.from_numpy(train_numpy).float()
        self.valid_tensor = torch.from_numpy(valid_numpy).float()
        self.test_tensor = torch.from_numpy(test_numpy).float()

        self.train_dataset = torch.utils.data.TensorDataset(self.train_tensor)
        self.valid_dataset = torch.utils.data.TensorDataset(self.valid_tensor)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_tensor)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, **self.kwargs)

        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            **self.kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            **self.kwargs)

        self.num_inputs = self.train_tensor.size()[1]


        self.modules = []
        for _ in range(self.num_blocks):
            self.modules += [
                MADE(self.num_inputs, self.num_hidden),
                BatchNormFlow(self.num_inputs),
                Reverse(self.num_inputs)
            ]

        self.model = FlowSequential(*self.modules)

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                module.bias.data.fill_(0)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)


    def flow_loss(self,u, log_jacob, size_average=True):
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        loss = -(log_probs + log_jacob).sum()
        if size_average:
            loss /= u.size(0)
        return loss


    def train(self,epoch):
        self.model.train()
        for batch_idx, data in enumerate(self.train_loader):
            if isinstance(data, list):
                data = data[0]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            u, log_jacob = self.model(data)
            loss = self.flow_loss(u, log_jacob)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

        for module in self.model.modules():
            if isinstance(module, BatchNormFlow):
                module.momentum = 0

        with torch.no_grad():
            self.model(self.train_loader.dataset.tensors[0].to(data.device))

        for module in self.model.modules():
            if isinstance(module, BatchNormFlow):
                module.momentum = 1


    def validate(self,epoch, model, loader, prefix='Validation'):
        self.model.eval()
        val_loss = 0

        for data in loader:
            if isinstance(data, list):
                data = data[0]
            data = data.to(self.device)
            with torch.no_grad():
                u, log_jacob = self.model(data)
                val_loss += self.flow_loss(
                    u, log_jacob, size_average=False).item()  # sum up batch loss

        val_loss /= len(loader.dataset)
        print('\n{} set: Average loss: {:.4f}\n'.format(prefix, val_loss))

        return val_loss

    def fit(self,epochs):

        best_validation_loss = float('inf')
        best_validation_epoch = 0
        best_model = self.model

        for epoch in range(epochs):
            self.train(epoch)
            validation_loss = self.validate(epoch, self.model, self.valid_loader)

            if epoch - best_validation_epoch >= 30:
                break

            if validation_loss < best_validation_loss:
                best_validation_epoch = epoch
                best_validation_loss = validation_loss
                best_model = copy.deepcopy(self.model)

            print('Best validation at epoch {}: Average loss: {:.4f}\n'.format(
                best_validation_epoch, best_validation_loss))

        self.validate(best_validation_epoch, best_model, self.test_loader, prefix='Test')

        self.model = best_model


    def plot(self):
        # generate some examples
        self.model.eval()
        u = np.random.randn(500, 2).astype(np.float32)
        u_tens = torch.from_numpy(u).to(self.device)
        x_synth = self.model.forward(u_tens, mode='inverse')[0].detach().cpu().numpy()

        import matplotlib.pyplot as plt

        valid = self.valid_tensor.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(valid[:,0], valid[:,1], '.')
        ax.set_title('Real data')

        ax = fig.add_subplot(122)
        ax.plot(x_synth[:,0], x_synth[:,1], '.')
        ax.set_title('Synth data')

        plt.show()







































def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, cond_in_features=None, bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self, num_inputs, num_hidden, num_cond_inputs=None, use_tanh=False):
        super(MADE, self).__init__()

        self.use_tanh = use_tanh

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)

        self.trunk = nn.Sequential(
            nn.ReLU(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask),
            nn.ReLU(),
            nn.MaskedLinear(num_hidden, num_inputs * 2, output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            if self.use_tanh:
                a = torch.tanh(a)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                if self.use_tanh:
                    a = torch.tanh(a)
                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)

class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets
